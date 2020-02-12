from unittest import mock

import pytest

from libertem.io.cache.stats import CacheStats, CacheItem


@pytest.fixture
def cs():
    cs = CacheStats()
    with cs:
        yield cs


@pytest.fixture
def ci():
    return CacheItem(
        dataset="deadbeef",
        partition=5,
        size=768,
        path="/tmp/dont_care",
    )


def test_first_miss(cs, ci):
    with mock.patch('libertem.io.cache.stats.time.time', side_effect=lambda: 42):
        cs.record_miss(ci)

    active_items = cs.get_active_items()
    assert len(active_items) == 1
    si = list(active_items)[0]
    assert si.partition == 5
    assert si.last_access == 42
    assert si.hits == 0


def test_first_hits(cs, ci):
    with mock.patch('libertem.io.cache.stats.time.time', side_effect=[21, 42]):
        cs.record_miss(ci)
        cs.record_hit(ci)

    active_items = cs.get_active_items()
    assert len(active_items) == 1
    si = list(active_items)[0]
    assert si.partition == 5
    assert si.last_access == 42
    assert si.hits == 1


def test_next_hits(cs, ci):
    with mock.patch('libertem.io.cache.stats.time.time', side_effect=[1, 2, 4, 8]):
        cs.record_miss(ci)
        cs.record_hit(ci)
        cs.record_hit(ci)
        cs.record_hit(ci)

    active_items = cs.get_active_items()
    assert len(active_items) == 1
    si = list(active_items)[0]
    assert si.partition == 5
    assert si.last_access == 8
    assert si.hits == 3


def test_eviction(cs, ci):
    with mock.patch('time.time', side_effect=[1, 2, 4]):
        cs.record_miss(ci)
        cs.record_hit(ci)
        cs.record_eviction(ci)

    active_items = cs.get_active_items()
    assert len(active_items) == 0
    assert len(cs._items_to_remove) == 1
    assert len(cs._items_to_add) == 0
    assert len(cs._all_items.items()) == 1


def test_merge(cs, ci):
    pass
