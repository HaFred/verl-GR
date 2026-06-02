"""MiniOneRec trainer adapter."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.reward import extract_reward

from verl_gr.recipes.minionerec.minionerec_reward import ndcg_penalties, normalize_sid
from verl_gr.trainers.task_adapter import TrainerTaskAdapter
from verl_gr.workers.rollout.beam_config import (
    BEAM_RETURN_MODE_KEY,
    BEAM_SEARCH_PARAMS_KEY,
    BEAM_WIDTH_KEY,
)


class MiniOneRecTrainerAdapter(TrainerTaskAdapter):
    """MiniOneRec-specific trainer adapter."""

    def prepare_gen_batch(self, trainer, batch):
        return trainer._prepare_recommendation_gen_batch(batch)

    def postprocess_rewards(
        self,
        trainer,
        batch: DataProto,
        reward_batch: DataProto,
    ) -> tuple[DataProto, dict[str, Any]]:
        reward_tensor = reward_batch.batch["rm_scores"]
        if "responses" not in batch.batch or "reward_model" not in batch.non_tensor_batch:
            return reward_batch, {}

        completions = [normalize_sid(trainer.tokenizer.decode(ids, skip_special_tokens=True)) for ids in batch.batch["responses"]]
        targets = [normalize_sid(item.get("ground_truth", "")) for item in batch.non_tensor_batch["reward_model"]]
        group_keys = self._group_keys(batch)
        rule_rewards = np.array([float(pred == target and target != "") for pred, target in zip(completions, targets, strict=True)])
        ranking_rewards = np.zeros(len(completions), dtype=np.float32)
        group_has_hit = np.zeros(len(completions), dtype=np.float32)

        groups: dict[Any, list[int]] = defaultdict(list)
        for idx, key in enumerate(group_keys):
            groups[key].append(idx)

        for indices in groups.values():
            hit = any(rule_rewards[idx] > 0 for idx in indices)
            if not hit:
                continue
            group_has_hit[indices] = 1.0
            discounts = ndcg_penalties(len(indices))
            for local_rank, idx in enumerate(indices):
                if rule_rewards[idx] == 0:
                    ranking_rewards[idx] = discounts[local_rank]

        total_rewards = rule_rewards.astype(np.float32) + ranking_rewards
        reward_batch.batch["rm_scores"] = self._write_sequence_rewards(
            batch=batch,
            reward_tensor=reward_tensor,
            sequence_rewards=torch.tensor(total_rewards, dtype=reward_tensor.dtype, device=reward_tensor.device),
            pad_token_id=trainer.tokenizer.pad_token_id,
        )
        invalid_sid = np.array([float(not self._looks_like_sid(pred)) for pred in completions], dtype=object)
        return reward_batch, {
            "minionerec_rule_reward": rule_rewards.astype(object),
            "minionerec_ranking_reward": ranking_rewards.astype(object),
            "minionerec_group_has_hit": group_has_hit.astype(object),
            "minionerec_invalid_sid": invalid_sid,
        }

    def validate(self, trainer):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_ground_truths = []

        for test_data in trainer.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            val_kwargs = trainer.config.actor_rollout_ref.rollout.val_kwargs
            rollout_cfg = trainer.config.actor_rollout_ref.rollout
            rollout_custom = rollout_cfg.get("custom") or {}
            beam_width = int(rollout_custom.get(BEAM_WIDTH_KEY, val_kwargs.get("n", 1)))
            base_generations_per_prompt = int(
                rollout_custom.get("num_generations_per_prompt", max(1, int(rollout_cfg.get("n", 1)) // max(beam_width, 1)))
            )
            repeat_times = max(1, base_generations_per_prompt) * max(1, beam_width)
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)

            input_ids = test_batch.batch["input_ids"]
            if "raw_prompt" in test_batch.non_tensor_batch:
                sample_inputs.extend([str(v) for v in test_batch.non_tensor_batch["raw_prompt"]])
            else:
                sample_inputs.extend([trainer.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids])
            if "reward_model" in test_batch.non_tensor_batch:
                sample_ground_truths.extend(
                    [normalize_sid(item.get("ground_truth", "")) for item in test_batch.non_tensor_batch["reward_model"]]
                )

            test_gen_batch = trainer._prepare_recommendation_gen_batch(test_batch)
            meta_info = {
                **test_gen_batch.meta_info,
                "eos_token_id": trainer.tokenizer.eos_token_id,
                "pad_token_id": trainer.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": val_kwargs.do_sample,
                "validate": True,
                "global_steps": trainer.global_steps,
                BEAM_RETURN_MODE_KEY: "all_beams",
            }
            test_gen_batch.meta_info = meta_info

            size_divisor = (
                trainer.actor_rollout_wg.world_size
                if not trainer.async_rollout_mode
                else trainer.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not trainer.async_rollout_mode:
                test_output_gen_batch_padded = trainer.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = trainer.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            if trainer.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                trainer.checkpoint_manager.sleep_replicas()
                batch_reward = trainer._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                trainer.checkpoint_manager.update_weights(trainer.global_steps)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [normalize_sid(trainer.tokenizer.decode(ids, skip_special_tokens=True)) for ids in output_ids]
            sample_outputs.extend(output_texts)
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            reward_tensor, reward_extra_info = extract_reward(test_batch)
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                elif isinstance(values, list):
                    reward_extra_infos_dict[key].extend(values)
                else:
                    reward_extra_infos_dict[key].append(values)

            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    trainer.config.data.get("reward_fn_key", "data_source"),
                    test_batch.non_tensor_batch.get("source", test_batch.non_tensor_batch.get("data_source", ["minionerec"] * len(test_batch))),
                )
            )

        self.maybe_log_val_generations(trainer, sample_inputs, sample_outputs, sample_scores)
        val_data_dir = trainer.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self.dump_generations(
                trainer,
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                ground_truths=sample_ground_truths,
            )
        data_sources = np.concatenate(data_source_lst, axis=0) if data_source_lst else np.array(["minionerec"])
        metric_dict = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict.update(
            self._compute_pass_at_k_metrics(
                data_sources=data_sources,
                sample_inputs=sample_inputs,
                sample_outputs=sample_outputs,
                sample_ground_truths=sample_ground_truths,
                k=32,
            )
        )
        return metric_dict

    @staticmethod
    def _compute_pass_at_k_metrics(
        *,
        data_sources,
        sample_inputs: list[str],
        sample_outputs: list[str],
        sample_ground_truths: list[str],
        k: int,
    ) -> dict[str, float]:
        if not sample_outputs or not sample_ground_truths:
            return {}
        grouped_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, (data_source, prompt) in enumerate(zip(data_sources, sample_inputs, strict=True)):
            grouped_indices[(str(data_source), str(prompt))].append(idx)
        source_values: dict[str, list[float]] = defaultdict(list)
        for (data_source, _prompt), indices in grouped_indices.items():
            candidate_indices = indices[:k]
            gt_sid = normalize_sid(sample_ground_truths[indices[0]])
            hit = float(any(normalize_sid(sample_outputs[idx]) == gt_sid and gt_sid != "" for idx in candidate_indices))
            source_values[data_source].append(hit)
        metrics = {}
        for data_source, values in source_values.items():
            key = f"val-aux/{data_source}/pass_at_{k}"
            val = float(np.mean(values)) if values else 0.0
            metrics[key] = val
            metrics[f"{key}/mean"] = val
        return metrics

    @staticmethod
    def _group_keys(batch: DataProto) -> list[Any]:
        if "uid" in batch.non_tensor_batch:
            return [str(item) for item in batch.non_tensor_batch["uid"]]
        if "index" in batch.non_tensor_batch:
            return [str(item) for item in batch.non_tensor_batch["index"]]
        return [idx for idx in range(len(batch))]

    @staticmethod
    def _write_sequence_rewards(batch, reward_tensor, sequence_rewards, pad_token_id: int):
        rewritten = torch.zeros_like(reward_tensor)
        responses = batch.batch["responses"]
        response_mask = responses != pad_token_id
        valid_lengths = response_mask.sum(dim=1).clamp(min=1)
        rewritten[torch.arange(rewritten.size(0), device=rewritten.device), valid_lengths - 1] = sequence_rewards
        return rewritten

    @staticmethod
    def _looks_like_sid(text: str) -> bool:
        return text.startswith("<a_") and "<b_" in text and "<c_" in text


MiniOneRecTrainerHooks = MiniOneRecTrainerAdapter
