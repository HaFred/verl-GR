"""Rank-GRPO async rollout fast path.

The upstream single-turn agent loop issues one vLLM request per repeated rollout.
Rank-GRPO's prompts are text-only and repeated contiguously, so we can collapse
each prompt group into one vLLM request with ``n=<group size>`` and expand the
returned completions back into verl's normal DataProto layout.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import numpy as np
import ray
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    AsyncLLMServerManager,
)
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.utils import qwen2_5_vl_dedup_image_tokens
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    extract_prompt_logprobs,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica


class RankGRPOvLLMHttpServer(vLLMHttpServer):
    """vLLM server extension that returns all completions from a batched request."""

    async def generate_many(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        n: int,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        priority: int = 0,
    ) -> list[TokenOutput]:
        prompt_ids = normalize_token_ids(prompt_ids)
        sampling_params = dict(sampling_params)

        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            max_tokens = min(
                self.config.response_length,
                self.config.prompt_length + self.config.response_length - len(prompt_ids),
            )
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        sampling_params["n"] = max(1, int(n))
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling = SamplingParams(max_tokens=max_tokens, **sampling_params)

        prompt_ids = qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        lora_request = None
        if self.lora_as_adapter and VLLM_LORA_INT_ID in await self.engine.list_loras():
            lora_request = LoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
            )

        generator = self.engine.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data),
            sampling_params=sampling,
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )

        final_res = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        prompt_extra_fields = {"global_steps": self.global_steps}
        extract_prompt_logprobs(
            output=final_res,
            num_prompt_logprobs=sampling.prompt_logprobs,
            result_dict=prompt_extra_fields,
        )

        results: list[TokenOutput] = []
        for completion in final_res.outputs:
            token_ids = normalize_token_ids(completion.token_ids)
            log_probs = None
            if sampling.logprobs is not None and completion.logprobs is not None:
                log_probs = []
                for token_id, token_logprobs in zip(token_ids, completion.logprobs, strict=False):
                    entry = token_logprobs[token_id]
                    log_probs.append(entry.logprob if hasattr(entry, "logprob") else float(entry))

            finish_reason = completion.finish_reason
            if finish_reason == "abort":
                stop_reason = "aborted"
            elif finish_reason in ("stop", "length"):
                stop_reason = "completed"
            else:
                stop_reason = finish_reason

            results.append(
                TokenOutput(
                    token_ids=token_ids,
                    log_probs=log_probs,
                    routed_experts=getattr(completion, "routed_experts", None)
                    if self.config.enable_rollout_routing_replay
                    else None,
                    stop_reason=stop_reason,
                    num_preempted=getattr(completion, "num_preempted", None),
                    extra_fields=dict(prompt_extra_fields),
                )
            )
        return results


class RankGRPOvLLMReplica(vLLMReplica):
    """vLLM replica using the Rank-GRPO server extension."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(RankGRPOvLLMHttpServer)


class RankGRPOAsyncLLMServerManager(AsyncLLMServerManager):
    """Server manager with a batched token-in/token-out request method."""

    async def generate_many(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        n: int,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[TokenOutput]:
        server_id, server = await self._acquire_server(request_id)
        try:
            return await server.generate_many.remote(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                n=n,
                image_data=image_data,
                video_data=video_data,
                **kwargs,
            )
        finally:
            self._release_server(server_id)


class RankGRPOAgentLoopWorker(AgentLoopWorker):
    """Batch repeated Rank-GRPO single-turn rollouts before calling vLLM."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_manager = RankGRPOAsyncLLMServerManager(
            self.config,
            [(sid, self.server_manager._server_id_to_handle[sid]) for sid in self.server_manager._server_id_to_handle],
            load_balancer_handle=self.server_manager._load_balancer,
        )

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        if not self._can_use_rankgrpo_fast_path(batch):
            return await super().generate_sequences(batch)

        sampling_params = self._build_sampling_params(batch)
        groups = self._group_repeated_prompts(batch)

        tasks = [
            asyncio.create_task(self._generate_group(batch, positions, prompt_ids, sampling_params))
            for positions, prompt_ids in groups
        ]
        group_results = await asyncio.gather(*tasks)

        outputs_by_position: dict[int, Any] = {}
        for positions, outputs in group_results:
            if len(outputs) != len(positions):
                raise RuntimeError(f"vLLM returned {len(outputs)} completions for {len(positions)} Rank-GRPO rollouts.")
            for position, output in zip(positions, outputs, strict=True):
                outputs_by_position[position] = output

        outputs = [outputs_by_position[position] for position in range(len(batch))]
        return self._postprocess(
            outputs,
            input_non_tensor_batch=batch.non_tensor_batch,
            validate=batch.meta_info.get("validate", False),
        )

    def _can_use_rankgrpo_fast_path(self, batch: DataProto) -> bool:
        if self.processor is not None or self.reward_loop_worker_handles is not None or self.distillation_enabled:
            return False
        if "raw_prompt_ids" not in batch.non_tensor_batch:
            return False
        agent_names = batch.non_tensor_batch.get("agent_name")
        if agent_names is None:
            return True
        return all(str(name) == "single_turn_agent" for name in agent_names)

    def _build_sampling_params(self, batch: DataProto) -> dict[str, Any]:
        config = self.rollout_config
        sampling_params = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": 1.0,
            "logprobs": config.calculate_log_probs,
        }
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
        return sampling_params

    def _group_repeated_prompts(self, batch: DataProto) -> list[tuple[list[int], list[int]]]:
        groups: list[tuple[list[int], list[int]]] = []
        current_positions: list[int] = []
        current_key = None
        current_prompt_ids: list[int] | None = None

        for position, prompt_ids_value in enumerate(batch.non_tensor_batch["raw_prompt_ids"]):
            prompt_ids = normalize_token_ids(prompt_ids_value)
            key = tuple(prompt_ids)
            if current_key is not None and key != current_key:
                assert current_prompt_ids is not None
                groups.append((current_positions, current_prompt_ids))
                current_positions = []
            current_key = key
            current_prompt_ids = prompt_ids
            current_positions.append(position)

        if current_positions:
            assert current_prompt_ids is not None
            groups.append((current_positions, current_prompt_ids))
        return groups

    async def _generate_group(
        self,
        batch: DataProto,
        positions: list[int],
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> tuple[list[int], list[Any]]:
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            token_outputs = await self.server_manager.generate_many(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                n=len(positions),
            )

        outputs = []
        for token_output in token_outputs:
            response_ids = token_output.token_ids[: self.rollout_config.response_length]
            response_logprobs = (
                token_output.log_probs[: self.rollout_config.response_length]
                if token_output.log_probs is not None
                else None
            )
            output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=[1] * len(response_ids),
                response_logprobs=response_logprobs,
                routed_experts=token_output.routed_experts,
                multi_modal_data={},
                num_turns=2,
                metrics=AgentLoopMetrics(
                    generate_sequences=metrics["generate_sequences"],
                    num_preempted=token_output.num_preempted if token_output.num_preempted is not None else -1,
                ),
                extra_fields=dict(token_output.extra_fields or {}),
            )
            output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
            outputs.append(
                await self._agent_loop_postprocess(
                    output,
                    batch.meta_info.get("validate", False),
                    **{key: value[positions[0]] for key, value in batch.non_tensor_batch.items()},
                )
            )
        return positions, outputs


class RankGRPOAgentLoopManager(AgentLoopManager):
    """AgentLoopManager that keeps repeated Rank-GRPO rollout groups colocated."""

    def __init__(self, *args, **kwargs):
        self.rollout_replica_class = RankGRPOvLLMReplica
        self.agent_loop_workers_class = ray.remote(RankGRPOAgentLoopWorker)
        super().__init__(*args, **kwargs)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        if "raw_prompt_ids" not in prompts.non_tensor_batch:
            return await super().generate_sequences(prompts)

        chunk_indices = self._grouped_worker_indices(prompts)
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(prompts.select_idxs(indices))
                for worker, indices in zip(self.agent_loop_workers, chunk_indices, strict=False)
                if len(indices) > 0
            ]
        )
        output = DataProto.concat(outputs)

        metrics = [worker_output.meta_info.pop("metrics") for worker_output in outputs]
        timing = self._performance_metrics(metrics, output)
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _grouped_worker_indices(self, prompts: DataProto) -> list[list[int]]:
        groups: list[list[int]] = []
        current_group: list[int] = []
        current_key = None
        for idx, prompt_ids_value in enumerate(prompts.non_tensor_batch.get("raw_prompt_ids", [])):
            key = tuple(normalize_token_ids(prompt_ids_value))
            if current_key is not None and key != current_key:
                groups.append(current_group)
                current_group = []
            current_key = key
            current_group.append(idx)
        if current_group:
            groups.append(current_group)

        chunks = [[] for _ in self.agent_loop_workers]
        if not groups:
            return chunks

        # Preserve global sample order so DataProto.concat(worker_outputs) remains
        # aligned with the input batch without relying on an auxiliary field to
        # survive recipe-specific postprocessing.
        target_size = max(1, int(np.ceil(len(prompts) / max(1, len(chunks)))))
        chunk_idx = 0
        for group in groups:
            if chunks[chunk_idx] and len(chunks[chunk_idx]) + len(group) > target_size and chunk_idx < len(chunks) - 1:
                chunk_idx += 1
            chunks[chunk_idx].extend(group)
        return chunks
