"""Unit tests for lazy weight sync interval logic."""
import pytest


class MockWeightSyncCounter:
    def __init__(self, sync_interval: int = 4):
        self._weight_sync_interval = sync_interval
        self._steps_since_sync = 0
        self.sync_count = 0

    def should_sync(self) -> bool:
        self._steps_since_sync += 1
        if self._steps_since_sync < self._weight_sync_interval:
            return False
        self._steps_since_sync = 0
        return True

    def update_weights(self):
        if self.should_sync():
            self.sync_count += 1
            return True
        return False


def test_sync_every_n_steps():
    counter = MockWeightSyncCounter(sync_interval=4)
    for step in range(1, 13):
        synced = counter.update_weights()
        if step % 4 == 0:
            assert synced
        else:
            assert not synced
    assert counter.sync_count == 3


def test_sync_interval_1():
    counter = MockWeightSyncCounter(sync_interval=1)
    for step in range(1, 6):
        assert counter.update_weights()
    assert counter.sync_count == 5


def test_sync_counter_resets_after_sync():
    counter = MockWeightSyncCounter(sync_interval=3)
    assert not counter.update_weights()
    assert not counter.update_weights()
    assert counter.update_weights()
    assert not counter.update_weights()
    assert counter.sync_count == 1
