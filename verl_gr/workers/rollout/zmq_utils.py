"""ZMQ socket helpers for custom verl-GR rollout adapters."""

from __future__ import annotations

import os

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

ZMQ_SOCKET_PREFIX_ENV = "VERL_ZMQ_SOCKET_PREFIX"


def get_zmq_socket_prefix(namespace: str) -> str:
    return os.environ.get(ZMQ_SOCKET_PREFIX_ENV, f"verl-gr-{namespace}")


def build_zmq_handle(*, namespace: str, replica_rank: int | str, local_rank: int | str) -> str:
    prefix = get_zmq_socket_prefix(namespace)
    return f"ipc:///tmp/{prefix}-replica-{replica_rank}-rank-{local_rank}.sock"


class VerlGRVLLMColocateWorkerExtension(vLLMColocateWorkerExtension):
    """vLLM worker extension with a configurable ZMQ socket namespace."""

    def _get_zmq_handle(self) -> str:
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        namespace = os.environ.get("VERL_ROLLOUT_ZMQ_NAMESPACE", "vllm")
        return build_zmq_handle(namespace=namespace, replica_rank=replica_rank, local_rank=self.local_rank)
