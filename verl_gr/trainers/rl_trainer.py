"""RL trainer extensions for verl-GR with bridged ray-trainer API."""

import json
import math
import os
import shutil
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as RayPPOTrainerBase
from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager
from verl.trainer.ppo.reward import extract_reward
from verl.utils import tensordict_utils as tu
from verl.utils.torch_functional import masked_mean
from verl.workers.utils.padding import left_right_2_no_padding

from verl_gr.recipes.openonerec.onerec_trainer import (
    openonerec_evaluate_and_prune_checkpoint,
    openonerec_dump_generations,
    openonerec_maybe_log_val_generations,
    openonerec_validate,
)
from verl_gr.trainers.task_adapter import TrainerTaskAdapter
from verl_gr.workers.rollout.beam_config import (
    BEAM_RETURN_MODE_KEY,
    BEAM_SEARCH_PARAMS_KEY,
    BEAM_WIDTH_KEY,
    DECODE_CONFIG_KEY,
    build_two_stage_sampling_params,
    get_rollout_custom_nested_value,
)

AdvantageEstimator = getattr(core_algos, "AdvantageEstimator")
_RANKGRPO_TOKENIZER = None


class _OpenOneRecTrainerAdapter(TrainerTaskAdapter):
    def prepare_gen_batch(self, trainer, batch: DataProto) -> DataProto:
        return trainer._prepare_recommendation_gen_batch(batch)

    def validate(self, trainer):
        return openonerec_validate(trainer)

    def dump_generations(self, trainer, inputs, outputs, scores, reward_extra_infos_dict, dump_path, ground_truths=None):
        return openonerec_dump_generations(
            trainer,
            inputs=inputs,
            outputs=outputs,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=dump_path,
            ground_truths=ground_truths,
        )

    def maybe_log_val_generations(self, trainer, inputs, outputs, scores):
        return openonerec_maybe_log_val_generations(trainer, inputs=inputs, outputs=outputs, scores=scores)


class _RankGRPOTrainerAdapter(TrainerTaskAdapter):
    def prepare_gen_batch(self, trainer, batch: DataProto) -> DataProto:
        return trainer._prepare_recommendation_gen_batch(batch)

    def validate(self, trainer):
        return trainer._rankgrpo_validate()


def apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty: str = "kl"):
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)
    kld = kld * response_mask
    beta = kl_ctrl.value
    token_level_rewards = token_level_scores - beta * kld
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards
    return data, {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _rankgrpo_enabled(config: Any) -> bool:
    rank_cfg = _cfg_get(config, "rank_grpo", None)
    return bool(_cfg_get(rank_cfg, "enable", False))


def _decode_response_texts(responses: torch.Tensor, response_mask: torch.Tensor, tokenizer) -> list[str]:
    texts: list[str] = []
    for ids, mask in zip(responses, response_mask, strict=True):
        valid_ids = ids[mask.bool()].detach().cpu().tolist()
        texts.append(tokenizer.decode(valid_ids, skip_special_tokens=True))
    return texts


def _segment_rank_tokens(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    *,
    rank_separator: str,
    rec_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign natural rank ids to response tokens using newline-like separators."""

    device = responses.device
    batch_size, response_length = responses.size()
    seg_ids = torch.full((batch_size, response_length), -1, dtype=torch.long, device=device)
    try:
        separator_ids = tokenizer.encode(rank_separator, add_special_tokens=False)
    except Exception:
        separator_ids = []
    single_separator_id = int(separator_ids[0]) if len(separator_ids) == 1 else None

    for row_idx in range(batch_size):
        valid = int(response_mask[row_idx].sum().item())
        item_id = 0
        for token_idx in range(valid):
            seg_ids[row_idx, token_idx] = item_id
            token_id = int(responses[row_idx, token_idx].item())
            separator_count = 0
            if single_separator_id is not None and token_id == single_separator_id:
                separator_count = 1
            else:
                try:
                    piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False, skip_special_tokens=False)
                except TypeError:
                    piece = tokenizer.decode([token_id])
                except Exception:
                    piece = ""
                separator_count = str(piece).count(rank_separator)
            if separator_count > 0:
                item_id += separator_count

    rank_token_mask = response_mask.bool() & (seg_ids >= 0) & (seg_ids < rec_num)
    return seg_ids, rank_token_mask


def _compute_rank_grpo_advantage(
    data: DataProto,
    *,
    config,
    tokenizer,
    norm_adv_by_std_in_grpo: bool,
) -> DataProto:
    if tokenizer is None:
        raise ValueError("Rank-GRPO advantage computation requires the trainer tokenizer.")

    rank_cfg = _cfg_get(config, "rank_grpo", {}) or {}
    rec_num = int(_cfg_get(rank_cfg, "rec_num", 20))
    rank_separator = _cfg_get(rank_cfg, "rank_separator", "\n")
    year_tolerance = int(_cfg_get(rank_cfg, "year_tolerance", 2))
    exclude_seen = bool(_cfg_get(rank_cfg, "exclude_seen", True))
    normalize_by_std = bool(_cfg_get(rank_cfg, "normalize_by_std", norm_adv_by_std_in_grpo))

    from verl_gr.recipes.rankgrpo.rankgrpo_reward import rank_rewards_from_text

    responses = data.batch["responses"]
    response_mask = data.batch["response_mask"]
    response_texts = _decode_response_texts(responses, response_mask, tokenizer)
    reward_models = data.non_tensor_batch.get("reward_model")
    if reward_models is None:
        raise KeyError("Rank-GRPO requires `reward_model` in data.non_tensor_batch.")

    reward_rows = [
        rank_rewards_from_text(
            text,
            reward_model,
            rec_num=rec_num,
            year_tolerance=year_tolerance,
            exclude_seen=exclude_seen,
        )
        for text, reward_model in zip(response_texts, reward_models, strict=True)
    ]
    rank_rewards = torch.tensor(reward_rows, dtype=torch.float32, device=responses.device)

    uids = data.non_tensor_batch.get("uid")
    if uids is None:
        uids = list(range(rank_rewards.size(0)))
    uid_to_indices: dict[Any, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid_to_indices[uid].append(idx)

    rank_advantages = torch.zeros_like(rank_rewards)
    for indices in uid_to_indices.values():
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=responses.device)
        group_rewards = rank_rewards.index_select(0, idx_tensor)
        centered = group_rewards - group_rewards.mean(dim=0, keepdim=True)
        if normalize_by_std:
            std = group_rewards.std(dim=0, unbiased=False, keepdim=True)
            centered = centered / (std + 1e-4)
        rank_advantages.index_copy_(0, idx_tensor, centered)

    seg_ids, rank_token_mask = _segment_rank_tokens(
        responses,
        response_mask,
        tokenizer,
        rank_separator=rank_separator,
        rec_num=rec_num,
    )
    clamped_seg_ids = seg_ids.clamp(min=0, max=rec_num - 1)
    token_advantages = rank_advantages.gather(1, clamped_seg_ids)
    token_advantages = token_advantages * rank_token_mask.float()

    data.batch["advantages"] = token_advantages
    data.batch["returns"] = token_advantages
    data.batch["rank_token_mask"] = rank_token_mask
    data.batch["rank_seg_ids"] = seg_ids
    return data


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,  # noqa: ARG001
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    tokenizer=None,  # noqa: ARG001
) -> DataProto:
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        if _rankgrpo_enabled(config):
            if tokenizer is None:
                tokenizer = _RANKGRPO_TOKENIZER
            data = _compute_rank_grpo_advantage(
                data,
                config=config,
                tokenizer=tokenizer,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
        else:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RLTrainer(RayPPOTrainerBase):
    """RayPPOTrainer override with different workload helpers."""

    def __init__(self, *args, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        if tokenizer is None and len(args) >= 2:
            tokenizer = args[1]
        super().__init__(*args, **kwargs)
        global _RANKGRPO_TOKENIZER
        _RANKGRPO_TOKENIZER = tokenizer
        if _rankgrpo_enabled(self.config.algorithm):
            import verl.trainer.ppo.ray_trainer as ray_trainer_mod

            ray_trainer_mod.compute_advantage = compute_advantage

    def fit(self):
        logging_steps = self._as_int(_cfg_get(self.config.trainer, "logging_steps", 1), default=1)
        if logging_steps <= 1:
            return super().fit()

        from verl.utils.tracking import Tracking

        original_log = Tracking.log

        def log_every_n_steps(tracking_self, data, step, backend=None):
            if step == 0 or step % logging_steps == 0:
                return original_log(tracking_self, data=data, step=step, backend=backend)
            return None

        Tracking.log = log_every_n_steps
        try:
            return super().fit()
        finally:
            Tracking.log = original_log

    def _get_task_adapter(self) -> TrainerTaskAdapter:
        if hasattr(self, "_task_adapter"):
            return self._task_adapter

        task_name = str(_cfg_get(_cfg_get(self.config, "task", None), "name", "")).lower()
        if task_name == "openonerec":
            self._task_adapter = _OpenOneRecTrainerAdapter()
        elif task_name == "rankgrpo":
            self._task_adapter = _RankGRPOTrainerAdapter()
        else:
            self._task_adapter = TrainerTaskAdapter()
        return self._task_adapter

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _expected_actor_lr(self) -> float | None:
        """Best-effort actor LR for logging when the worker omits it."""

        optim_config = _cfg_get(self.config.actor_rollout_ref.actor, "optim", None)
        if optim_config is None:
            return None

        base_lr = self._as_float(_cfg_get(optim_config, "lr", None), default=-1.0)
        if base_lr < 0:
            return None

        total_steps = self._as_int(
            _cfg_get(optim_config, "total_training_steps", self.total_training_steps),
            default=self.total_training_steps,
        )
        if total_steps <= 0:
            total_steps = self.total_training_steps

        warmup_steps = self._as_int(_cfg_get(optim_config, "lr_warmup_steps", -1), default=-1)
        if warmup_steps <= 0:
            warmup_ratio = self._as_float(_cfg_get(optim_config, "lr_warmup_steps_ratio", 0.0), default=0.0)
            warmup_steps = int(warmup_ratio * total_steps)

        step = max(self._as_int(getattr(self, "global_steps", 0), default=0), 0)
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * float(step) / float(max(1, warmup_steps))

        scheduler_type = _cfg_get(optim_config, "lr_scheduler_type", _cfg_get(optim_config, "warmup_style", "constant"))
        if scheduler_type != "cosine":
            return base_lr

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
        min_lr_ratio = self._as_float(_cfg_get(optim_config, "min_lr_ratio", 0.0), default=0.0)
        num_cycles = self._as_float(_cfg_get(optim_config, "num_cycles", 0.5), default=0.5)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))
        return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_scale)

    def _add_actor_lr_metrics(self, metrics: dict[str, Any]) -> None:
        optim_config = _cfg_get(self.config.actor_rollout_ref.actor, "optim", None)
        if optim_config is not None and "actor/base_lr" not in metrics:
            base_lr = self._as_float(_cfg_get(optim_config, "lr", None), default=-1.0)
            if base_lr >= 0:
                metrics["actor/base_lr"] = base_lr

        if "actor/lr" in metrics:
            return
        if "lr" in metrics:
            metrics["actor/lr"] = metrics["lr"]
            return

        expected_lr = self._expected_actor_lr()
        if expected_lr is not None:
            metrics["actor/lr"] = expected_lr

    def _update_actor(self, batch: DataProto) -> DataProto:
        actor_output = super()._update_actor(batch)
        self._add_actor_lr_metrics(actor_output.meta_info["metrics"])
        return actor_output

    def _compute_eval_actor_metrics(self, batch: DataProto) -> dict[str, Any]:
        """Compute actor loss metrics in eval mode without stepping the optimizer."""

        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch.meta_info["temperature"] = rollout_config.temperature

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            compute_loss=True,
            global_batch_size=batch.batch.batch_size[0],
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        return dict(tu.get(output, "metrics") or {})

    @staticmethod
    def _mean_metric(values: list[tuple[float, int]]) -> float | None:
        total_weight = sum(weight for _, weight in values)
        if total_weight <= 0:
            return None
        return sum(value * weight for value, weight in values) / total_weight

    def _rankgrpo_validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        eval_loss_values: list[tuple[float, int]] = []

        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            test_batch = test_batch.repeat(repeat_times=val_kwargs.n, interleave=True)
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            self.checkpoint_manager.sleep_replicas()
            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            test_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            reward_tensor, reward_extra_info = extract_reward(test_batch)
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                reward_extra_infos_dict.setdefault(key, [])
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            if "response_mask" not in test_batch.batch.keys():
                test_batch.batch["response_mask"] = compute_response_mask(test_batch)
            test_batch.meta_info["global_token_num"] = torch.sum(test_batch.batch["attention_mask"], dim=-1).tolist()

            old_log_prob, _ = self._compute_old_log_prob(test_batch)
            old_log_prob.batch.pop("entropys", None)
            test_batch = test_batch.union(old_log_prob)
            if self.use_reference_policy:
                ref_log_prob = self._compute_ref_log_prob(test_batch)
                test_batch = test_batch.union(ref_log_prob)

            test_batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_info:
                test_batch.non_tensor_batch.update({key: np.array(value) for key, value in reward_extra_info.items()})
            if self.config.algorithm.use_kl_in_reward:
                test_batch, _ = apply_kl_penalty(
                    test_batch,
                    kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
            else:
                test_batch.batch["token_level_rewards"] = test_batch.batch["token_level_scores"]

            test_batch = compute_advantage(
                test_batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                config=self.config.algorithm,
                tokenizer=self.tokenizer,
            )
            eval_actor_metrics = self._compute_eval_actor_metrics(test_batch)
            eval_loss = eval_actor_metrics.get("loss")
            if eval_loss is not None and math.isfinite(self._as_float(eval_loss, default=float("nan"))):
                eval_loss_values.append((float(eval_loss), int(reward_tensor.shape[0])))
            self.checkpoint_manager.update_weights(self.global_steps)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                ground_truths=sample_gts,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        metric_dict = self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
        eval_loss = self._mean_metric(eval_loss_values)
        if eval_loss is not None:
            metric_dict["eval/loss"] = eval_loss
        return metric_dict

    def _checkpoint_topk_state_path(self) -> str:
        return os.path.join(self.config.trainer.default_local_dir, "topk_checkpoints.json")

    def _load_topk_checkpoint_state(self) -> list[dict[str, Any]]:
        if hasattr(self, "_topk_checkpoints"):
            return self._topk_checkpoints
        state_path = self._checkpoint_topk_state_path()
        try:
            with open(state_path) as f:
                state = json.load(f)
        except FileNotFoundError:
            state = []
        self._topk_checkpoints = state if isinstance(state, list) else []
        return self._topk_checkpoints

    def _save_topk_checkpoint_state(self, state: list[dict[str, Any]]) -> None:
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
        with open(self._checkpoint_topk_state_path(), "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        if state:
            latest_kept_step = max(int(entry["step"]) for entry in state)
            latest_path = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
            with open(latest_path, "w") as f:
                f.write(str(latest_kept_step))
        self._topk_checkpoints = state

    def _select_topk_metric(self, metrics: dict[str, Any]) -> tuple[str | None, float | None]:
        metric_name = _cfg_get(
            self.config.trainer,
            "best_ckpt_metric",
            _cfg_get(self.config.trainer, "topk_ckpt_metric", None),
        )
        if metric_name:
            value = metrics.get(metric_name)
            return metric_name, self._as_float(value, default=float("nan"))

        for candidate in (
            "val-core/rankgrpo/reward/mean@1",
            "val-core/rankgrpo/score/mean@1",
            "val-core/rankgrpo/rank_reward_sum/mean@1",
        ):
            if candidate in metrics:
                return candidate, self._as_float(metrics[candidate], default=float("nan"))

        for key in sorted(metrics):
            if key.startswith("val-core/") and key.endswith("/mean@1"):
                return key, self._as_float(metrics[key], default=float("nan"))
        return None, None

    def _update_topk_checkpoints(self, metrics: dict[str, Any]) -> None:
        prune_enabled = _cfg_get(self.config.trainer, "best_ckpt_prune_enable", True)
        if isinstance(prune_enabled, str):
            prune_enabled = prune_enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
        if not prune_enabled:
            return

        top_k = self._as_int(
            _cfg_get(
                self.config.trainer,
                "best_ckpts_to_keep",
                _cfg_get(self.config.trainer, "topk_ckpt_keep", 0),
            ),
            default=0,
        )
        if top_k <= 0 or self.global_steps <= 0:
            return

        ckpt_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.isdir(ckpt_dir):
            return

        metric_name, metric_value = self._select_topk_metric(metrics)
        if metric_name is None or metric_value is None or not math.isfinite(metric_value):
            print("[topk] No finite validation metric found; skipping checkpoint ranking.")
            return

        mode = str(
            _cfg_get(
                self.config.trainer,
                "best_ckpt_mode",
                _cfg_get(self.config.trainer, "topk_ckpt_mode", "max"),
            )
        ).lower()
        reverse = mode != "min"
        state = [entry for entry in self._load_topk_checkpoint_state() if int(entry.get("step", -1)) != self.global_steps]
        state.append(
            {
                "step": int(self.global_steps),
                "metric": metric_name,
                "value": float(metric_value),
                "path": ckpt_dir,
            }
        )
        state.sort(key=lambda entry: float(entry["value"]), reverse=reverse)
        keep = state[:top_k]
        drop = state[top_k:]

        keep_paths = {entry["path"] for entry in keep}
        for entry in drop:
            path = entry.get("path")
            if path and path not in keep_paths and os.path.isdir(path):
                shutil.rmtree(path)
                print(f"[topk] Removed checkpoint outside top-{top_k}: {path}")

        self._save_topk_checkpoint_state(keep)
        print(f"[topk] Kept top-{top_k} checkpoints by {metric_name}: {keep}")

    @staticmethod
    def _ensure_reward_routing_keys(proto: DataProto) -> None:
        """Ensure both source aliases exist for reward-loop compatibility."""
        non_tensor = proto.non_tensor_batch
        if "data_source" not in non_tensor and "source" in non_tensor:
            non_tensor["data_source"] = non_tensor["source"]
        if "source" not in non_tensor and "data_source" in non_tensor:
            non_tensor["source"] = non_tensor["data_source"]

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        return self._get_task_adapter().prepare_gen_batch(self, batch)

    def _prepare_recommendation_gen_batch(self, batch: DataProto) -> DataProto:
        """Prepare generation batch without conflicting prompt tensors.

        In verl>=0.7.1 async rollout mode, generation output may include input_ids.
        If original training batch still carries prompt-side input_ids/attention_mask/
        position_ids, DataProto.union() asserts on key collisions. For OneRec dataset,
        we remove those prompt tensors before generation and keep reward-routing keys.
        """
        reward_keys = set(
            {
                "source",
                "data_source",
                "reward_model",
                "uid",
                "raw_prompt",
                "multi_modal_data",
                "tools_kwargs",
                "interaction_kwargs",
            }
        ) & batch.non_tensor_batch.keys()
        batch_keys_to_pop = [
            key for key in ("input_ids", "attention_mask", "position_ids") if key in batch.batch.keys()
        ]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        self._ensure_reward_routing_keys(gen_batch)
        rollout_cfg = self.config.actor_rollout_ref.rollout
        if rollout_cfg.get("name") == "two_stage":
            rollout_custom = rollout_cfg.get("custom") or {}
            reasoning_max_tokens = rollout_custom.get(
                "stage1_max_tokens",
                get_rollout_custom_nested_value(
                    rollout_cfg,
                    (DECODE_CONFIG_KEY, "reasoning", "max_tokens"),
                    self.config.data.get("max_response_length", rollout_cfg.response_length),
                ),
            )
            beam_width = rollout_custom.get(
                BEAM_WIDTH_KEY,
                rollout_custom.get("stage2_beam_size", 32),
            )
            item_max_tokens = rollout_custom.get(
                "stage2_num_tokens",
                get_rollout_custom_nested_value(
                    rollout_cfg,
                    (BEAM_SEARCH_PARAMS_KEY, "max_tokens"),
                    3,
                ),
            )
            gen_batch.meta_info.update(
                {
                    "enable_two_stage_rollout": True,
                    "max_tokens": self.config.data.get("max_response_length", rollout_cfg.response_length),
                }
            )
            beam_search_params = rollout_custom.get(BEAM_SEARCH_PARAMS_KEY) or {}
            if beam_search_params.get("constraint") is not None:
                gen_batch.meta_info["constraint"] = beam_search_params.get("constraint")
            gen_batch.meta_info.update(
                build_two_stage_sampling_params(
                    reasoning_max_tokens=int(reasoning_max_tokens),
                    item_max_tokens=int(item_max_tokens),
                    beam_width=int(beam_width),
                )
            )
        elif rollout_cfg.get("name") == "constrained_beam":
            rollout_custom = rollout_cfg.get("custom") or {}
            beam_search_params = rollout_custom.get(BEAM_SEARCH_PARAMS_KEY) or {}
            beam_width = int(rollout_custom.get(BEAM_WIDTH_KEY, rollout_custom.get("beam_size", 20)))
            item_max_tokens = int(beam_search_params.get("max_tokens", self.config.data.get("max_response_length", 64)))
            gen_batch.meta_info.update(
                {
                    "enable_constrained_beam_rollout": True,
                    "max_tokens": item_max_tokens,
                    BEAM_WIDTH_KEY: beam_width,
                    BEAM_RETURN_MODE_KEY: "best_only",
                    BEAM_SEARCH_PARAMS_KEY: dict(beam_search_params),
                }
            )
            if beam_search_params.get("constraint") is not None:
                gen_batch.meta_info["constraint"] = beam_search_params.get("constraint")
        return gen_batch

    def _validate(self):
        metrics = self._get_task_adapter().validate(self)
        self._update_topk_checkpoints(metrics)
        return metrics

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path, ground_truths=None):
        return self._get_task_adapter().dump_generations(
            self,
            inputs=inputs,
            outputs=outputs,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=dump_path,
            ground_truths=ground_truths,
        )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        return self._get_task_adapter().maybe_log_val_generations(self, inputs=inputs, outputs=outputs, scores=scores)

    def _save_checkpoint(self):
        super()._save_checkpoint()
        task_name = str(_cfg_get(_cfg_get(self.config, "task", None), "name", "")).lower()
        if task_name != "openonerec":
            return
        local_global_step_folder = f"{self.config.trainer.default_local_dir}/global_step_{self.global_steps}"
        openonerec_evaluate_and_prune_checkpoint(
            self,
            local_global_step_folder,
            metrics=getattr(self, "_last_validation_metrics", None),
        )

