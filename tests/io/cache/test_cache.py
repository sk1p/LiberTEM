from unittest import mock
import os

import pytest

from libertem.io.cache.stats import CacheStats, CacheItem
from libertem.io.cache.base import CacheException
from libertem.io.cache.cache import Cache
from libertem.io.cache.strategy import LRUCacheStrategy


@pytest.fixture
def cache():
    cs = CacheStats()
    strategy = LRUCacheStrategy(capacity=1024)
    cache = Cache(stats=cs, strategy=strategy)
    yield cache


@pytest.fixture
def cachedir(tmpdir_factory):
    yield tmpdir_factory.mktemp('cache')


def _make_item(item: CacheItem):
    assert not os.path.exists(item.path)
    with open(item.path, "w") as f:
        f.truncate(item.size)


def test_default_strategy(cache, cachedir):
    # cache items for first dataset:
    ci1 = CacheItem(
        dataset="deadbeef1",
        partition=5,
        size=512,
        path=os.path.join(cachedir, 'ci1')
    )
    ci2 = CacheItem(
        dataset="deadbeef1",
        partition=6,
        size=512,
        path=os.path.join(cachedir, 'ci2')
    )
    for i in [ci1, ci2]:
        # should be added without evicting anything:
        assert len(cache.evict(i.dataset, i.size)) == 0
        _make_item(i)
        cache.record_miss(i)

    assert len(list(cache._stats.get_active_items())) == 2

    # for second dataset:
    ci4 = CacheItem(
        dataset="deadbeef2",
        partition=1,
        size=256,
        path=os.path.join(cachedir, 'ci4')
    )
    ds2_items = [ci4]
    for i in ds2_items:
        assert os.path.exists(
            cache._stats.get_item(dataset="deadbeef1", partition=5).path
        )
        evicted = cache.evict(i.dataset, i.size)
        assert len(evicted) == 1
        assert not os.path.exists(evicted[0].path)
        assert not os.path.exists(
            cache._stats.get_item(dataset="deadbeef1", partition=5).path
        )
        assert evicted[0].dataset == "deadbeef1"
        assert evicted[0].partition == 5
        _make_item(i)
        cache.record_miss(i)

    assert len(list(cache._stats.get_active_items())) == 2


def test_cache_too_small_err_1(cache, cachedir):
    # cache items for first dataset:
    ci1 = CacheItem(
        dataset="deadbeef1",
        partition=5,
        size=1025,
        path=os.path.join(cachedir, 'ci1')
    )
    with pytest.raises(CacheException) as e:
        cache.evict(ci1.dataset, ci1.size)
    assert e.match("not enough cache capacity")


def test_cache_too_small_err_2(cache, cachedir):
    # cache items for first dataset:
    ci1 = CacheItem(
        dataset="deadbeef1",
        partition=5,
        size=512,
        path=os.path.join(cachedir, 'ci1')
    )
    ci2 = CacheItem(
        dataset="deadbeef1",
        partition=6,
        size=512,
        path=os.path.join(cachedir, 'ci2')
    )
    ci3 = CacheItem(
        dataset="deadbeef1",
        partition=7,
        size=512,
        path=os.path.join(cachedir, 'ci3')
    )
    assert len(cache.evict(ci1.dataset, ci1.size)) == 0
    _make_item(ci1)
    cache.record_miss(ci1)
    assert len(cache.evict(ci2.dataset, ci2.size)) == 0
    _make_item(ci2)
    cache.record_miss(ci2)
    with pytest.raises(CacheException) as e:
        cache.evict(ci3.dataset, ci3.size)
    assert e.match("not enough cache capacity")


def test_cache_exactly_large_enough_1(cache, cachedir):
    # cache items for first dataset:
    ci1 = CacheItem(
        dataset="deadbeef1",
        partition=5,
        size=1024,
        path=os.path.join(cachedir, 'ci1')
    )
    assert len(cache.evict(ci1.dataset, ci1.size)) == 0


def test_cache_exactly_large_enough_2(cache, cachedir):
    ci1 = CacheItem(
        dataset="deadbeef1",
        partition=5,
        size=512,
        path=os.path.join(cachedir, 'ci1')
    )
    assert len(cache.evict(ci1.dataset, ci1.size)) == 0
    ci2 = CacheItem(
        dataset="deadbeef1",
        partition=6,
        size=512,
        path=os.path.join(cachedir, 'ci2')
    )
    assert len(cache.evict(ci2.dataset, ci2.size)) == 0


def test_item_already_evicted(cache, cachedir):
    """
    The cache may have incomplete information - there may be other tasks running
    that change the filesystem, but the cache stats are only updated at the
    beginning of the task.

    In this case, we test the eviction rules: the strategy must return all eligible
    items, because some of them may already be deleted (for example evited by
    another task, or manual cleanup).
    """

    # all regular here:
    ci2 = CacheItem(
        dataset="deadbeef1",
        partition=6,
        size=768,
        path=os.path.join(cachedir, 'ci2')
    )
    assert len(cache.evict(ci2.dataset, ci2.size)) == 0
    _make_item(ci2)
    cache.record_miss(ci2)

    ci1 = CacheItem(
        dataset="deadbeef1",
        partition=5,
        size=256,
        path=os.path.join(cachedir, 'ci1')
    )
    assert len(cache.evict(ci1.dataset, ci1.size)) == 0
    _make_item(ci1)
    cache.record_miss(ci1)

    # now, someone deletes ci2:
    os.unlink(ci2.path)

    # let's see who gets evicted if we try to fill in another item:
    ci3 = CacheItem(
        dataset="deadbeef2",
        partition=1,
        size=256,
        path=os.path.join(cachedir, 'ci4')
    )
    evicted = cache.evict(ci3.dataset, ci3.size)
    assert len(evicted) == 1
    # should be partition 5, as partition 6 was already evicted elsewhere:
    assert evicted[0].dataset == "deadbeef1"
    assert evicted[0].partition == 5
