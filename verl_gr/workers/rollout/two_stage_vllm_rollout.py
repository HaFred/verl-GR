"""OpenOneRec-style two-stage vLLM rollout."""

from __future__ import annotations

import logging
import time
from importlib import import_module
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.ray_utils import get_event_loop

from verl_gr.third_party.vllm import BeamSearchParams, LoRARequest
from verl_gr.workers.rollout.primitives import (
    PreparedPromptInputs,
    build_lora_requests,
    build_sampling_params,
    expand_beam_candidates,
    pack_rollout_batch,
    prepare_prompt_token_inputs,
)

_vllm_rollout_module = None
_server_adapter_cls = None
_vllm_rollout_import_errors: list[str] = []

# Candidate layouts across forks/versions:
# - OpenOneRec/legacy: verl.workers.rollout.vllm_rollout.vllm_rollout_spmd
# - Some forks:         verl.workers.rollout.vllm_rollout (re-export)
# - Newer layouts:      verl.workers.rollout.vllm_rollout.vllm_rollout
for _path in (
    "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd",
    "verl.workers.rollout.vllm_rollout.vllm_rollout",
    "verl.workers.rollout.vllm_rollout",
):
    try:
        _mod = import_module(_path)
        _srv = getattr(_mod, "ServerAdapter", None)
        if _srv is not None and _server_adapter_cls is None:
            _server_adapter_cls = _srv
        _cls = getattr(_mod, "vLLMRollout", None)
        if _cls is not None:
            _vllm_rollout_module = _mod
            vLLMRollout = _cls
            break
        _vllm_rollout_import_errors.append(f"{_path}: missing symbol vLLMRollout")
    except Exception as exc:  # pragma: no cover - env-dependent import probe
        _vllm_rollout_import_errors.append(f"{_path}: {exc.__class__.__name__}: {exc}")

if _vllm_rollout_module is None and _server_adapter_cls is None:
    raise ImportError(f"Cannot find vLLM rollout runtime symbols. Probed: {' | '.join(_vllm_rollout_import_errors)}")

RolloutBaseCls = vLLMRollout if _vllm_rollout_module is not None else _server_adapter_cls
_USING_SERVER_ADAPTER_BACKEND = _vllm_rollout_module is None

_pre_process_inputs = getattr(_vllm_rollout_module, "_pre_process_inputs", None) if _vllm_rollout_module else None
if _pre_process_inputs is None:

    def _pre_process_inputs(pad_token_id: int, prompt_token_ids: torch.Tensor) -> list[int]:
        non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
        return prompt_token_ids[non_pad_index:].tolist()

logger = logging.getLogger(__name__)


def resolve_rollout_n(*, prompts: DataProto, kwargs: dict[str, Any], config: Any) -> int:
    """Rollout group size (GRPO `actor_rollout_ref.rollout.n`), distinct from vLLM sample `n`."""
    if "rollout_n" in prompts.meta_info:
        return int(prompts.meta_info["rollout_n"])
    if "rollout_n" in kwargs:
        return int(kwargs["rollout_n"])
    return int(getattr(config, "n", 1) or 1)


def repeat_interleave_layout(batch_size: int, rollout_n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Layout from `DataProto.repeat(..., interleave=True)` (verl `repeat_interleave` on dim 0)."""
    if rollout_n <= 1 or batch_size % rollout_n != 0:
        return None
    positions = np.arange(batch_size, dtype=np.int64)
    repeat_k = positions % rollout_n
    base_idx = positions // rollout_n
    return repeat_k, base_idx


def _uids_match_interleaved_repeat(uid: np.ndarray, rollout_n: int) -> bool:
    """True when each block of `rollout_n` rows shares the same uid (GRPO repeat batch)."""
    if len(uid) % rollout_n != 0:
        return False
    n_grp = len(uid) // rollout_n
    for g in range(n_grp):
        block = uid[g * rollout_n : (g + 1) * rollout_n]
        if not np.all(block == block[0]):
            return False
    return True


def _compare_vanilla_flag(config: Any) -> bool:
    v = getattr(config, "compare_vanilla_vs_stage1_reuse", None)
    if v is None and hasattr(config, "get"):
        v = config.get("compare_vanilla_vs_stage1_reuse", False)
    return bool(v)


def _build_single_stage2_vllm_input(
    vllm_row: dict[str, Any],
    cot_output: Any,
    *,
    prefix_ids: list[int],
    vocab_size: int,
) -> dict[str, Any]:
    """One stage-2 vLLM prompt: original tokens + CoT + item prefix marker."""
    if hasattr(cot_output, "outputs"):
        cot_token_ids = list(cot_output.outputs[0].token_ids)
    else:
        cot_token_ids = list(cot_output)
    cot_token_ids_filtered = [tid for tid in cot_token_ids if tid < vocab_size]
    original_prompt_ids = vllm_row["prompt_token_ids"]
    new_prompt_ids = list(original_prompt_ids) + cot_token_ids_filtered + prefix_ids
    stage2_input: dict[str, Any] = {"prompt_token_ids": new_prompt_ids}
    if "multi_modal_data" in vllm_row:
        stage2_input["multi_modal_data"] = vllm_row["multi_modal_data"]
    return stage2_input


def _response_token_counts(dp: DataProto, pad_token_id: int) -> np.ndarray:
    r = dp.batch["responses"]
    return (r != pad_token_id).sum(dim=-1).detach().cpu().numpy().astype(np.float64)


def _rollout_group_quality_metrics(lens: np.ndarray, rollout_n: int) -> tuple[float, float]:
    """Mean within-group variance of response lengths; mean within-group distinctness (hash prefix)."""
    if rollout_n <= 1 or len(lens) % rollout_n != 0:
        return 0.0, 1.0
    n_grp = len(lens) // rollout_n
    vars_: list[float] = []
    distinct: list[float] = []
    for g in range(n_grp):
        sl = lens[g * rollout_n : (g + 1) * rollout_n]
        vars_.append(float(np.var(sl)))
        # cheap distinctness: discretize lengths + first "bin" only (no tensor access here)
        distinct.append(float(len(np.unique(sl.astype(np.int64)))) / float(rollout_n))
    return float(np.mean(vars_)), float(np.mean(distinct))


class Stage2vLLMRollout(RolloutBaseCls):
    """Stage-2 only: beam-search item outputs given precomputed stage-1 (CoT) completions.

    Beam search follows highest-probability expansions and is **deterministic** for a fixed model
    and prompt. For GRPO-style diversity with a shared CoT, use stochastic sampling (see feature
    rollout in :class:`TwoStagevLLMRollout`).
    """

    @torch.no_grad()
    def _stage2_generation(
        self,
        prompts: DataProto,
        cot_outputs: list[Any],
        *,
        prepared_inputs: PreparedPromptInputs,
        **kwargs: Any,
    ) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        if len(cot_outputs) != batch_size:
            raise ValueError(
                f"cot_outputs length ({len(cot_outputs)}) must match prompt batch ({batch_size})."
            )

        vllm_inputs = prepared_inputs.vllm_inputs
        non_tensor_batch = prepared_inputs.non_tensor_batch

        stage2_inputs = []
        tokenizer = self._get_tokenizer()
        prefix_ids = tokenizer.encode("\n<|sid_begin|>", add_special_tokens=False)
        vocab_size = len(tokenizer)

        for i, output in enumerate(cot_outputs):
            stage2_inputs.append(
                _build_single_stage2_vllm_input(
                    vllm_inputs[i],
                    output,
                    prefix_ids=prefix_ids,
                    vocab_size=vocab_size,
                )
            )

        beam_width = kwargs.get("stage2_beam_size", getattr(self.config, "stage2_beam_size", 32))
        max_tokens_item = kwargs.get(
            "stage2_max_tokens",
            kwargs.get("stage2_num_tokens", getattr(self.config, "stage2_num_tokens", 16)),
        )
        if BeamSearchParams is None:
            raise ImportError("BeamSearchParams not available; cannot run stage-2 beam search.")

        beam_params = BeamSearchParams(beam_width=beam_width, max_tokens=max_tokens_item)
        item_outputs = self.inference_engine.beam_search(prompts=stage2_inputs, params=beam_params)

        expansion = expand_beam_candidates(
            item_outputs=item_outputs,
            stage_inputs=stage2_inputs,
            idx=idx,
            attention_mask=attention_mask,
            position_ids=position_ids,
            non_tensor_batch=non_tensor_batch,
            beam_width=beam_width,
            return_all_beams=kwargs.get("return_all_beams", True),
            beam_idxs=non_tensor_batch.get("beam_idx"),
        )

        return pack_rollout_batch(
            idx=expansion.idx,
            responses=expansion.responses,
            attention_mask=expansion.attention_mask,
            position_ids=expansion.position_ids,
            pad_token_id=self.pad_token_id,
            eos_token_id=eos_token_id,
            response_length=self.config.response_length,
            calculate_log_probs=self.config.calculate_log_probs,
            non_tensor_batch=expansion.non_tensor_batch,
        )


class TwoStagevLLMRollout(RolloutBaseCls):
    """Generate CoT first, then beam-search item outputs."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        tokenizer = getattr(self, "tokenizer", None) or getattr(getattr(self, "model_config", None), "tokenizer", None)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
        self._supports_native_engine = hasattr(self, "inference_engine")

    def _server_generate_tokens(
        self,
        *,
        prompt_token_ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None = None,
        stop: list[str] | None = None,
        include_stop_str_in_output: bool | None = None,
        multi_modal_data: dict[str, Any] | None = None,
    ) -> list[int]:
        sampling_params: dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "n": 1,
        }
        if seed is not None:
            sampling_params["seed"] = int(seed) % (2**31)
        if stop is not None:
            sampling_params["stop"] = stop
        if include_stop_str_in_output is not None:
            sampling_params["include_stop_str_in_output"] = include_stop_str_in_output
        image_data = None
        video_data = None
        if isinstance(multi_modal_data, dict):
            image_data = multi_modal_data.get("image")
            video_data = multi_modal_data.get("video")
        request_id = f"two_stage_{time.time_ns()}"
        loop = get_event_loop()
        out = loop.run_until_complete(
            self._execute_method(
                "generate",
                args=(prompt_token_ids, sampling_params, request_id, image_data, video_data, 0),
            )
        )
        if out is None:
            return []
        return list(getattr(out, "token_ids", []))

    def _get_tokenizer(self) -> Any:
        if self._supports_native_engine:
            return self.inference_engine.get_tokenizer()
        tok = getattr(getattr(self, "model_config", None), "tokenizer", None)
        if tok is None:
            raise RuntimeError("Tokenizer is not available on TwoStage rollout.")
        return tok

    @torch.no_grad()
    def _beam_stage2_suffix_tokens(
        self,
        stage2_input: dict[str, Any],
        *,
        beam_width: int,
        max_tokens_item: int,
        lora_requests: list[Any] | None,
    ) -> list[int]:
        """Deterministic beam item continuation (suffix after stage-2 prompt)."""
        if not self._supports_native_engine:
            # verl_v0.7.1 async server path has no beam_search API; approximate with greedy decode.
            return self._server_generate_tokens(
                prompt_token_ids=list(stage2_input["prompt_token_ids"]),
                max_tokens=max_tokens_item,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                multi_modal_data=stage2_input.get("multi_modal_data"),
            )
        if BeamSearchParams is None:
            raise ImportError("BeamSearchParams not available; cannot run stage-2 beam search.")
        beam_params = BeamSearchParams(beam_width=beam_width, max_tokens=max_tokens_item)
        item_outputs = self.inference_engine.beam_search(prompts=[stage2_input], params=beam_params)
        out = item_outputs[0]
        prompt_len = len(stage2_input["prompt_token_ids"])
        num_seqs = len(out.sequences)
        if num_seqs < 1:
            raise RuntimeError("beam_search returned no sequences")
        best_seq = out.sequences[0]
        toks = getattr(best_seq, "tokens", None) or getattr(best_seq, "token_ids", None)
        if toks is None:
            raise RuntimeError("beam_search sequence has no tokens/token_ids")
        return list(toks[prompt_len:])

    @torch.no_grad()
    def _sample_stage2_suffix_tokens(
        self,
        stage2_input: dict[str, Any],
        *,
        max_tokens_item: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        lora_requests: list[Any] | None,
    ) -> list[int]:
        """Stochastic single completion for shared-CoT follow-up rollouts (GRPO diversity)."""
        if not self._supports_native_engine:
            return self._server_generate_tokens(
                prompt_token_ids=list(stage2_input["prompt_token_ids"]),
                max_tokens=max_tokens_item,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                multi_modal_data=stage2_input.get("multi_modal_data"),
            )
        sp = build_sampling_params(
            max_tokens=max_tokens_item,
            n=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            extra_kwargs={"seed": int(seed) % (2**31)},
        )
        one_out = self.inference_engine.generate(
            prompts=[stage2_input],
            sampling_params=sp,
            lora_request=lora_requests,
            use_tqdm=False,
        )[0]
        prompt_len = len(stage2_input["prompt_token_ids"])
        token_ids = list(one_out.outputs[0].token_ids)
        gen = token_ids[prompt_len:] if len(token_ids) >= prompt_len else token_ids
        tokenizer = self._get_tokenizer()
        vocab_size = len(tokenizer)
        return [tid for tid in gen if tid < vocab_size]

    @torch.no_grad()
    def _feature_rollout_hybrid_grpo(
        self,
        prompts: DataProto,
        rollout_n: int,
        **kwargs: Any,
    ) -> DataProto:
        """One stage-1 per GRPO group; row0 stage-2 beam; rows 1..n-1 stage-2 sample with distinct seeds."""
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)
        num_groups = batch_size // rollout_n

        first_row_indices = [g * rollout_n for g in range(num_groups)]
        prompts_stage1 = prompts.select_idxs(first_row_indices)
        prepared_first = prepare_prompt_token_inputs(
            prompts_stage1,
            pad_token_id=self.pad_token_id,
            preprocess_inputs=_pre_process_inputs,
        )
        cot_per_group = self._run_stage1_cot_outputs(prompts_stage1, prepared_first, **kwargs)
        prepared_full = prepare_prompt_token_inputs(
            prompts,
            pad_token_id=self.pad_token_id,
            preprocess_inputs=_pre_process_inputs,
        )

        tokenizer = self.inference_engine.get_tokenizer()
        prefix_ids = tokenizer.encode("\n<|sid_begin|>", add_special_tokens=False)
        vocab_size = len(tokenizer)

        beam_width = int(kwargs.get("stage2_beam_size", getattr(self.config, "stage2_beam_size", 32)))
        max_tokens_item = int(
            kwargs.get(
                "stage2_max_tokens",
                kwargs.get("stage2_num_tokens", getattr(self.config, "stage2_num_tokens", 16)),
            )
        )
        temperature = float(kwargs.get("temperature", 1.0))
        if temperature < 1e-4:
            temperature = 1e-4
        top_p = float(kwargs.get("top_p", 1.0))
        top_k = int(kwargs.get("top_k", -1))

        step_hint = prompts.meta_info.get("global_steps")
        try:
            step_int = int(step_hint) if step_hint is not None else 0
        except (TypeError, ValueError):
            step_int = 0

        responses: list[list[int]] = []
        for g in range(num_groups):
            row0 = g * rollout_n
            vllm_row = prepared_full.vllm_inputs[row0]
            stage2_input = _build_single_stage2_vllm_input(
                vllm_row,
                cot_per_group[g],
                prefix_ids=prefix_ids,
                vocab_size=vocab_size,
            )
            beam_suffix = self._beam_stage2_suffix_tokens(
                stage2_input,
                beam_width=beam_width,
                max_tokens_item=max_tokens_item,
                lora_requests=None,
            )
            responses.append(beam_suffix)

            for k in range(1, rollout_n):
                seed = step_int * 1_000_003 + g * 17_011 + k * 499 + rollout_n
                sample_suffix = self._sample_stage2_suffix_tokens(
                    stage2_input,
                    max_tokens_item=max_tokens_item,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    lora_requests=None,
                )
                responses.append(sample_suffix)

        non_tensor_batch = dict(prompts.non_tensor_batch)
        return pack_rollout_batch(
            idx=idx,
            responses=responses,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pad_token_id=self.pad_token_id,
            eos_token_id=eos_token_id,
            response_length=self.config.response_length,
            calculate_log_probs=self.config.calculate_log_probs,
            non_tensor_batch=non_tensor_batch,
        )

    @torch.no_grad()
    def _run_stage1_cot_outputs(
        self,
        prompts: DataProto,
        prepared_inputs: PreparedPromptInputs,
        **kwargs: Any,
    ) -> list[Any]:
        vllm_inputs = prepared_inputs.vllm_inputs

        stage1_max_tokens = kwargs.get(
            "stage1_max_tokens",
            getattr(self.config, "stage1_max_tokens", kwargs.get("max_tokens", 1024)),
        )
        temperature = float(kwargs.get("temperature", 1.0))
        top_p = float(kwargs.get("top_p", 1.0))
        top_k = int(kwargs.get("top_k", -1))
        if self._supports_native_engine:
            batch_size = len(vllm_inputs)
            cot_sampling_params = build_sampling_params(
                max_tokens=stage1_max_tokens,
                n=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=["</think>"],
                include_stop_str_in_output=True,
            )
            lora_requests = build_lora_requests(
                self.inference_engine,
                lora_kwargs=self.lora_kwargs,
                lora_request_cls=LoRARequest,
                batch_size=batch_size,
            )
            return self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=cot_sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

        outs: list[list[int]] = []
        for row in vllm_inputs:
            outs.append(
                self._server_generate_tokens(
                    prompt_token_ids=list(row["prompt_token_ids"]),
                    max_tokens=int(stage1_max_tokens),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=["</think>"],
                    include_stop_str_in_output=True,
                    multi_modal_data=row.get("multi_modal_data"),
                )
            )
        return outs

    @torch.no_grad()
    def _two_stage_generation(self, prompts: DataProto, **kwargs: Any) -> DataProto:
        prepared_inputs = prepare_prompt_token_inputs(
            prompts,
            pad_token_id=self.pad_token_id,
            preprocess_inputs=_pre_process_inputs,
        )
        cot_outputs = self._run_stage1_cot_outputs(prompts, prepared_inputs, **kwargs)
        if not self._supports_native_engine:
            idx = prompts.batch["input_ids"]
            attention_mask = prompts.batch["attention_mask"]
            position_ids = prompts.batch["position_ids"]
            eos_token_id = prompts.meta_info["eos_token_id"]
            tokenizer = self._get_tokenizer()
            prefix_ids = tokenizer.encode("\n<|sid_begin|>", add_special_tokens=False)
            vocab_size = len(tokenizer)
            max_tokens_item = int(
                kwargs.get(
                    "stage2_max_tokens",
                    kwargs.get("stage2_num_tokens", getattr(self.config, "stage2_num_tokens", 16)),
                )
            )
            stage2_temp = float(kwargs.get("temperature", 1.0))
            stage2_top_p = float(kwargs.get("top_p", 1.0))
            stage2_top_k = int(kwargs.get("top_k", -1))
            responses: list[list[int]] = []
            for i, row in enumerate(prepared_inputs.vllm_inputs):
                stage2_input = _build_single_stage2_vllm_input(
                    row,
                    cot_outputs[i],
                    prefix_ids=prefix_ids,
                    vocab_size=vocab_size,
                )
                responses.append(
                    self._server_generate_tokens(
                        prompt_token_ids=list(stage2_input["prompt_token_ids"]),
                        max_tokens=max_tokens_item,
                        temperature=stage2_temp,
                        top_p=stage2_top_p,
                        top_k=stage2_top_k,
                        multi_modal_data=stage2_input.get("multi_modal_data"),
                    )
                )
            return pack_rollout_batch(
                idx=idx,
                responses=responses,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pad_token_id=self.pad_token_id,
                eos_token_id=eos_token_id,
                response_length=self.config.response_length,
                calculate_log_probs=self.config.calculate_log_probs,
                non_tensor_batch=dict(prompts.non_tensor_batch),
            )
        return Stage2vLLMRollout._stage2_generation(
            self, prompts, cot_outputs, prepared_inputs=prepared_inputs, **kwargs
        )

    @staticmethod
    def _build_rollout_cmp_metrics(
        *,
        out_default: DataProto,
        out_feature: DataProto,
        default_s: float,
        feature_s: float,
        rollout_n: int,
        pad_token_id: int,
    ) -> dict[str, float]:
        ld = _response_token_counts(out_default, pad_token_id)
        lf = _response_token_counts(out_feature, pad_token_id)
        vd, dd = _rollout_group_quality_metrics(ld, rollout_n)
        vf, df = _rollout_group_quality_metrics(lf, rollout_n)
        total_tok_d = float(ld.sum())
        total_tok_f = float(lf.sum())
        return {
            "default_wall_s": float(default_s),
            "feature_wall_s": float(feature_s),
            "default_resp_tokens_total": total_tok_d,
            "feature_resp_tokens_total": total_tok_f,
            "default_throughput_resp_tok_per_s": float(total_tok_d / max(default_s, 1e-9)),
            "feature_throughput_resp_tok_per_s": float(total_tok_f / max(feature_s, 1e-9)),
            "default_mean_within_group_resp_len_var": vd,
            "feature_mean_within_group_resp_len_var": vf,
            "default_mean_within_group_len_distinct_frac": dd,
            "feature_mean_within_group_len_distinct_frac": df,
            "pairwise_mean_abs_resp_len_diff": float(np.mean(np.abs(ld - lf))),
        }

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs: Any) -> DataProto:
        for key in [
            "max_tokens",
            "temperature",
            "n",
            "top_p",
            "top_k",
            "stage1_max_tokens",
            "stage2_beam_size",
            "stage2_max_tokens",
            "stage2_num_tokens",
            "return_all_beams",
            "rollout_n",
            "stage2_sample_seed",
        ]:
            if key in prompts.meta_info:
                kwargs[key] = prompts.meta_info[key]

        batch_size = len(prompts.batch)
        rollout_n = resolve_rollout_n(prompts=prompts, kwargs=kwargs, config=self.config)
        layout = repeat_interleave_layout(batch_size, rollout_n)

        uid = prompts.non_tensor_batch.get("uid")
        if layout is not None and uid is not None and not _uids_match_interleaved_repeat(uid, rollout_n):
            layout = None

        compare = _compare_vanilla_flag(self.config)

        if compare and layout is not None:
            t0 = time.perf_counter()
            out_default = self._two_stage_generation(prompts, **kwargs)
            t_def = time.perf_counter() - t0
            t0 = time.perf_counter()
            out_feature = self._feature_rollout_hybrid_grpo(prompts, rollout_n, **kwargs)
            t_feat = time.perf_counter() - t0
            cmp = self._build_rollout_cmp_metrics(
                out_default=out_default,
                out_feature=out_feature,
                default_s=t_def,
                feature_s=t_feat,
                rollout_n=rollout_n,
                pad_token_id=self.pad_token_id,
            )
            meta = dict(out_default.meta_info) if out_default.meta_info else {}
            meta["openonerec_rollout_cmp"] = cmp
            meta["openonerec_cmp_feature_rollout"] = out_feature
            out_default.meta_info = meta
            logger.info(
                "openonerec_rollout_cmp step=%s default_s=%.4f feature_s=%.4f "
                "default_len_var=%.4f feature_len_var=%.4f tok/s_def=%.1f tok/s_feat=%.1f",
                prompts.meta_info.get("global_steps"),
                cmp["default_wall_s"],
                cmp["feature_wall_s"],
                cmp["default_mean_within_group_resp_len_var"],
                cmp["feature_mean_within_group_resp_len_var"],
                cmp["default_throughput_resp_tok_per_s"],
                cmp["feature_throughput_resp_tok_per_s"],
            )
            return out_default

        return self._two_stage_generation(prompts, **kwargs)
