import glob
import os

from libertem.io.cache.stats import CacheStats, CacheItem
from libertem.io.cache.strategy import CacheStrategy


class Cache:
    def __init__(self, stats: CacheStats, strategy: CacheStrategy):
        """
        Cache object, to be used on a worker node. The interface used by `Partition`\\ s
        to manage the cache. May directly remove files, directories, etc.
        """
        self._stats = stats
        self.strategy = strategy

    def record_hit(self, cache_item: CacheItem):
        with self._stats:
            self._stats.record_hit(cache_item)

    def record_miss(self, cache_item: CacheItem):
        with self._stats:
            self._stats.record_miss(cache_item)

    def evict(self, cache_key: str, size: int):
        """
        Make place for `size` bytes which will be used
        by the dataset identified by the `cache_key`.
        """

        if self.strategy.sufficient_space_for(size, self._stats):
            return []

        evicted = []
        with self._stats:
            candidates = self.strategy.get_victim_list(cache_key, size, self._stats)
            for cache_item in candidates:
                # if item has been deleted by the user or another task
                # don't add to evicted list, as the space may already
                # be assumed free by another task! do record eviction,
                # to not leave files behind in any case
                self._stats.record_eviction(cache_item)
                if os.path.exists(cache_item.path):
                    os.unlink(cache_item.path)
                else:
                    continue
                evicted.append(cache_item)

                if sum(i.size for i in evicted) >= size:
                    break
        return evicted

    def collect_orphans(self, base_path: str):
        """
        Check the filesystem structure and record all partitions
        that are missing in the db as orphans, to be deleted on demand.
        """
        raise NotImplementedError("to be reimplemented")
        # the structure here is: {base_path}/{dataset_cache_key}/parts/*
        orphans = []
        with self._stats:
            for path in glob.glob(os.path.join(base_path, "*", "parts", "*")):
                size = os.stat(path).st_size
                res = self._stats.maybe_orphan(OrphanItem(path=path, size=size))
                if res is not None:
                    orphans.append(res)
        return orphans
