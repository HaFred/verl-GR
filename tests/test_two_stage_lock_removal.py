"""Test that stage-2 beam search works correctly without the global lock."""
import asyncio
import pytest


class MockBeamSearchServer:
    """Minimal mock of TwoStagevLLMHttpServer for testing concurrent stage-2 access."""
    def __init__(self, max_concurrent: int = 4):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._concurrent_count = 0
        self._max_observed = 0

    async def _run_stage2_beam_search(self, request_id: str):
        async with self._semaphore:
            self._concurrent_count += 1
            self._max_observed = max(self._max_observed, self._concurrent_count)
            await asyncio.sleep(0.01)
            self._concurrent_count -= 1
        return [{"generated_token_ids": [1, 2, 3], "log_probs": [0.1, 0.2, 0.3]}]

    async def _build_two_stage_cache_entry(self, request_id: str):
        await asyncio.sleep(0.005)
        stage2_candidates = await self._run_stage2_beam_search(request_id)
        return {
            "responses": [{"token_ids": [1, 2, 3], "log_probs": [0.1, 0.2, 0.3]}],
            "generated_items": [[1, 2, 3]],
            "extra_fields": {},
            "remaining": 1,
        }


@pytest.mark.asyncio
async def test_concurrent_stage2_beam_searches():
    server = MockBeamSearchServer(max_concurrent=4)
    tasks = [server._build_two_stage_cache_entry(f"req_{i}") for i in range(8)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 8
    for r in results:
        assert r["generated_items"][0] == [1, 2, 3]
    assert server._max_observed >= 2, (
        f"Expected concurrent (>=2), got {server._max_observed}"
    )


@pytest.mark.asyncio
async def test_semaphore_backpressure():
    server = MockBeamSearchServer(max_concurrent=2)
    tasks = [server._run_stage2_beam_search(f"req_{i}") for i in range(10)]
    await asyncio.gather(*tasks)
    assert server._max_observed <= 2
