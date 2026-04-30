"""OpenOneRec-specific async agent loop extensions for two-stage rollout."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import numpy as np
import ray

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    RolloutTraceConfig,
    get_trajectory_info,
    register,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput
from verl_gr.workers.rollout.beam_config import (
    BEAM_GROUP_ID_KEY,
    BEAM_INDEX_KEY,
    BEAM_WIDTH_KEY,
    build_two_stage_sampling_params,
    get_rollout_custom_nested_value,
    get_rollout_custom_value,
)


@register("openonerec_two_stage_agent")
class OpenOneRecTwoStageAgentLoop(SingleTurnAgentLoop):
    """Single-turn agent loop that routes repeated samples into two-stage groups."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        sampling_params = dict(sampling_params)

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        prompt_ids = await self.apply_chat_template(messages, images=images, videos=videos)

        beam_width = max(1, int(sampling_params.get(BEAM_WIDTH_KEY, 1)))
        rollout_n = int(kwargs.get("trajectory_rollout_n", 0))
        stage1_sample_idx = rollout_n // beam_width
        beam_index = rollout_n % beam_width
        sample_index = kwargs.get("trajectory_sample_index", -1)
        step = kwargs.get("trajectory_step", -1)
        validate = int(bool(kwargs.get("trajectory_validate", False)))

        sampling_params["stage1_sample_idx"] = stage1_sample_idx
        sampling_params[BEAM_INDEX_KEY] = beam_index
        sampling_params[BEAM_GROUP_ID_KEY] = f"{step}:{validate}:{sample_index}:{stage1_sample_idx}"
        request_id = sampling_params[BEAM_GROUP_ID_KEY]

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        result = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        extra_info = kwargs.get("extra_info")
        if extra_info is None:
            extra_info = {}
        else:
            extra_info = dict(extra_info)
        result.extra_fields["extra_info"] = extra_info
        if "generated_items" in result.extra_fields:
            result.extra_fields["extra_info"]["generated_items"] = result.extra_fields["generated_items"]

        result.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return result


class OpenOneRecAgentLoopWorker(AgentLoopWorker):
    """Custom worker that injects two-stage rollout params without patching verl."""

    _nvml_module = None
    _nvml_handles: list[Any] | None = None
    _nvml_init_failed = False

    def __init__(self, *args, **kwargs):
        # Import side effect ensures the custom agent loop is registered on the worker.
        import verl_gr.recipes.openonerec.two_stage_agent_loop  # noqa: F401

        super().__init__(*args, **kwargs)

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _get_nvml_handles() -> list[Any] | None:
        if OpenOneRecAgentLoopWorker._nvml_init_failed:
            return None
        if OpenOneRecAgentLoopWorker._nvml_handles is not None:
            return OpenOneRecAgentLoopWorker._nvml_handles

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            OpenOneRecAgentLoopWorker._nvml_module = pynvml
            OpenOneRecAgentLoopWorker._nvml_handles = handles
            return handles
        except Exception:
            OpenOneRecAgentLoopWorker._nvml_init_failed = True
            return None

    @staticmethod
    def _read_gpu_utilization() -> float | None:
        handles = OpenOneRecAgentLoopWorker._get_nvml_handles()
        pynvml = OpenOneRecAgentLoopWorker._nvml_module
        if not handles or pynvml is None:
            return None

        gpu_utils: list[float] = []
        for handle in handles:
            try:
                gpu_utils.append(float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu))
            except Exception:
                continue
        if not gpu_utils:
            return None

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            selected = []
            for token in visible_devices.split(","):
                token = token.strip()
                if token.isdigit():
                    idx = int(token)
                    if 0 <= idx < len(gpu_utils):
                        selected.append(gpu_utils[idx])
            if selected:
                return sum(selected) / len(selected)
        return sum(gpu_utils) / len(gpu_utils)

    async def _read_server_runtime_metrics(self) -> dict[str, int] | None:
        server_handles = getattr(self.server_manager, "server_handles", None)
        if not server_handles:
            return None

        metrics_calls = []
        for server in server_handles:
            method = getattr(server, "get_two_stage_runtime_metrics", None)
            if method is not None:
                metrics_calls.append(method.remote())
        if not metrics_calls:
            return None

        metrics_results = await asyncio.gather(*metrics_calls, return_exceptions=True)
        aggregate = {
            "max_inflight_engine_requests": 0,
            "inflight_engine_requests": 0,
            "engine_request_waiters": 0,
            "pending_build_tasks": 0,
            "cache_entries": 0,
        }
        healthy_results = 0
        for result in metrics_results:
            if isinstance(result, Exception):
                continue
            healthy_results += 1
            for key in aggregate:
                aggregate[key] += int(result.get(key, 0))

        if healthy_results == 0:
            return None
        return aggregate

    async def generate_sequences(self, batch):
        if self.rollout_config.name != "two_stage":
            return await super().generate_sequences(batch)

        config = self.rollout_config
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
        if batch.meta_info.get("max_tokens") is not None:
            sampling_params["max_tokens"] = batch.meta_info["max_tokens"]

        sampling_params["enable_two_stage_rollout"] = True
        reasoning_max_tokens = batch.meta_info.get(
            "decode_config",
            {},
        ).get(
            "reasoning",
            {},
        ).get(
            "max_tokens",
            get_rollout_custom_nested_value(
                config,
                ("decode_config", "reasoning", "max_tokens"),
                get_rollout_custom_value(config, "stage1_max_tokens", config.response_length),
            ),
        )
        beam_width = int(
            batch.meta_info.get(
                BEAM_WIDTH_KEY,
                get_rollout_custom_value(
                    config,
                    BEAM_WIDTH_KEY,
                    get_rollout_custom_value(config, "stage2_beam_size", 32),
                ),
            )
        )
        item_max_tokens = int(
            batch.meta_info.get(
                "beam_search_params",
                {},
            ).get(
                "max_tokens",
                get_rollout_custom_nested_value(
                    config,
                    ("beam_search_params", "max_tokens"),
                    get_rollout_custom_value(config, "stage2_num_tokens", 16),
                ),
            )
        )
        sampling_params.update(
            build_two_stage_sampling_params(
                reasoning_max_tokens=int(reasoning_max_tokens),
                item_max_tokens=item_max_tokens,
                beam_width=beam_width,
                return_all_beams=batch.meta_info.get("beam_return_mode") == "all_beams",
            )
        )

        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["openonerec_two_stage_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        def build_task_kwargs(task_idx: int) -> dict[str, Any]:
            kwargs = {key: value[task_idx] for key, value in batch.non_tensor_batch.items()}
            kwargs["trajectory_step"] = trajectory_info[task_idx]["step"]
            kwargs["trajectory_sample_index"] = trajectory_info[task_idx]["sample_index"]
            kwargs["trajectory_rollout_n"] = trajectory_info[task_idx]["rollout_n"]
            kwargs["trajectory_validate"] = trajectory_info[task_idx]["validate"]
            return kwargs

        async def run_indexed_task(task_idx: int, task_kwargs: dict[str, Any]):
            trace_sample = task_idx in traced_indices
            result = await self._run_agent_loop(
                sampling_params,
                trajectory_info[task_idx],
                trace=trace_sample,
                **task_kwargs,
            )
            return task_idx, result

        if batch.meta_info.get("validate", False):
            total_tasks = len(batch)
            started_at = time.monotonic()
            outputs = [None] * total_tasks
            max_concurrent_requests = int(get_rollout_custom_value(config, "validation_max_concurrent_requests", 256))
            min_concurrent_requests = int(get_rollout_custom_value(config, "validation_min_concurrent_requests", beam_width))
            adaptive_concurrency = self._to_bool(
                get_rollout_custom_value(config, "validation_adaptive_concurrency", False)
            )
            target_gpu_util = float(get_rollout_custom_value(config, "validation_target_gpu_utilization", 85.0))
            gpu_util_tolerance = float(get_rollout_custom_value(config, "validation_gpu_util_tolerance", 7.5))
            concurrency_step = int(get_rollout_custom_value(config, "validation_concurrency_step", beam_width))

            if max_concurrent_requests <= 0:
                max_concurrent_requests = total_tasks
            max_concurrent_requests = max(1, min(max_concurrent_requests, total_tasks))
            min_concurrent_requests = max(1, min(min_concurrent_requests, max_concurrent_requests))
            concurrency_step = max(1, concurrency_step)

            def normalize_chunk_size(raw_size: int, *, pending: int) -> int:
                if pending <= 0:
                    return 0
                size = max(1, min(raw_size, pending))
                if pending <= beam_width:
                    return pending
                if beam_width > 1:
                    size = max(beam_width, (size // beam_width) * beam_width)
                    size = min(size, pending)
                return max(1, size)

            chunk_size = normalize_chunk_size(max_concurrent_requests, pending=total_tasks)
            print(
                "[Validation Progress] Starting async two-stage rollout: "
                f"total_requests={total_tasks}, max_concurrent_requests={chunk_size}, "
                f"adaptive_concurrency={adaptive_concurrency}"
            )

            completed_count = 0
            chunk_start = 0
            while chunk_start < total_tasks:
                chunk_end = min(chunk_start + chunk_size, total_tasks)
                tasks = [
                    asyncio.create_task(run_indexed_task(task_idx, build_task_kwargs(task_idx)))
                    for task_idx in range(chunk_start, chunk_end)
                ]
                for task in asyncio.as_completed(tasks):
                    output_idx, output = await task
                    outputs[output_idx] = output
                    completed_count += 1
                    if completed_count % 100 == 0 or completed_count == total_tasks:
                        elapsed = time.monotonic() - started_at
                        rate = completed_count / elapsed if elapsed > 0 else 0.0
                        print(
                            "[Validation Progress] "
                            f"completed={completed_count}/{total_tasks}, elapsed={elapsed:.1f}s, rate={rate:.2f} req/s"
                        )
                pending_after_chunk = total_tasks - completed_count
                if adaptive_concurrency and pending_after_chunk > 0:
                    next_chunk = chunk_size
                    server_metrics = await self._read_server_runtime_metrics()
                    if server_metrics is not None:
                        inflight = int(server_metrics["inflight_engine_requests"])
                        waiters = int(server_metrics["engine_request_waiters"])
                        pending_build_tasks = int(server_metrics["pending_build_tasks"])
                        max_inflight = max(1, int(server_metrics["max_inflight_engine_requests"]))
                        inflight_ratio = inflight / max_inflight
                        if waiters > 0 or pending_build_tasks > 0 or inflight_ratio >= 0.95:
                            next_chunk -= concurrency_step
                        elif inflight_ratio <= 0.70:
                            next_chunk += concurrency_step
                    else:
                        sampled_gpu_util = self._read_gpu_utilization()
                        if sampled_gpu_util is not None:
                            if sampled_gpu_util < target_gpu_util - gpu_util_tolerance:
                                next_chunk += concurrency_step
                            elif sampled_gpu_util > target_gpu_util + gpu_util_tolerance:
                                next_chunk -= concurrency_step
                    next_chunk = max(min_concurrent_requests, min(max_concurrent_requests, next_chunk))
                    chunk_size = normalize_chunk_size(next_chunk, pending=pending_after_chunk)
                    if server_metrics is not None:
                        print(
                            "[Validation Adaptive] "
                            f"inflight={server_metrics['inflight_engine_requests']}/"
                            f"{server_metrics['max_inflight_engine_requests']}, "
                            f"waiters={server_metrics['engine_request_waiters']}, "
                            f"pending_build={server_metrics['pending_build_tasks']}, "
                            f"pending={pending_after_chunk}, next_chunk_size={chunk_size}"
                        )
                    else:
                        sampled_gpu_util = self._read_gpu_utilization()
                        gpu_util_text = f"{sampled_gpu_util:.1f}%" if sampled_gpu_util is not None else "n/a"
                        print(
                            "[Validation Adaptive] "
                            f"gpu_util={gpu_util_text}, pending={pending_after_chunk}, next_chunk_size={chunk_size}"
                        )
                chunk_start = chunk_end
        else:
            tasks = [
                asyncio.create_task(run_indexed_task(task_idx, build_task_kwargs(task_idx)))
                for task_idx in range(len(batch))
            ]
            indexed_outputs = await asyncio.gather(*tasks)
            outputs = [None] * len(indexed_outputs)
            for output_idx, output in indexed_outputs:
                outputs[output_idx] = output

        output = self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch)
        return output


class OpenOneRecAgentLoopManager(AgentLoopManager):
    """Manager that swaps in the OpenOneRec worker implementation."""

    agent_loop_workers_class = ray.remote(OpenOneRecAgentLoopWorker)
