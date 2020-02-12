import types
from typing import Union, Set
from collections import defaultdict
import datetime
import sqlite3
import time


class CacheItem:
    """
    A CacheItem describes a single unit of data that is cached, in this case
    a partition of the CachedDataSet.
    """
    def __init__(self, dataset: str, partition: int, size: int, path: str):
        """
        Parameters
        ----------
        dataset
            dataset id string, for example the cache key
        partition
            partition index
        size
            on-disk partition size in bytes
        path
            full absolute path to the file for the partition
        """
        # dataset and partition are read-only, as they form the identity of this object
        self._dataset = dataset
        self._partition = partition
        self.size = size
        self.path = path

    @property
    def dataset(self):
        return self._dataset

    @property
    def partition(self):
        return self._partition

    def __eq__(self, other):
        # dataset and partition are composite pk
        return self.dataset == other.dataset and self.partition == other.partition

    def __hash__(self):
        # invariant: a == b => hash(a) == hash(b)
        return hash(self.key())

    def key(self):
        """
        get a uniquely identifying key for this cache item
        """
        return (self.dataset, self.partition)

    def __repr__(self):
        return "<CacheItem: %s/%d>" % (self.dataset, self.partition)


class StatsItem(CacheItem):
    def __init__(self, hits: int, last_access: float, *args, **kwargs):
        """
        Parameters
        ----------
        hits
            number of recorded cache hits
        last_access
            utc timestamp of last access, use 0 if not known
        """
        self.hits = hits
        self.last_acchess = last_access
        super().__init__(*args, **kwargs)

    def hit(self):
        self.hits += 1
        return self.hits

    def miss(self):
        self.hits = 0
        return self.hits

    def set_last_access(self, timestamp):
        self.last_access = timestamp


class CacheStats:
    def __init__(self):
        self._evictions = defaultdict(lambda: 0)
        self._all_items = {}
        self._active_items = set()
        self._items_to_add = set()
        self._items_to_remove = set()

    def get_item(self, dataset, partition):
        return self._all_items[(dataset, partition)]

    def _get_timestamp(self, timestamp) -> float:
        if timestamp is None:
            return time.time()
        return timestamp

    def _stats_item(self, cache_item: CacheItem) -> StatsItem:
        """
        Get or create the StatsItem that matches the given CacheItem
        """
        return self._all_items.get(
            cache_item.key(),
            StatsItem(
                dataset=cache_item.dataset,
                partition=cache_item.partition,
                size=cache_item.size,
                path=cache_item.path,
                hits=0,
                last_access=0,
            )
        )

    def _add_item(self, stats_item: StatsItem):
        self._all_items[stats_item.key()] = stats_item
        self._items_to_add.add(stats_item)
        self._active_items.add(stats_item)

    def _remove_item(self, stats_item: CacheItem):
        self._all_items[stats_item.key()] = stats_item
        self._items_to_remove.add(stats_item)
        self._items_to_add -= self._items_to_remove
        if stats_item in self._active_items:
            self._active_items.remove(stats_item)

    def record_hit(self, cache_item: CacheItem, timestamp=None):
        """
        increment hit counter, update last access timestamp
        """
        stats_item = self._stats_item(cache_item)
        timestamp = self._get_timestamp(timestamp)
        stats_item.set_last_access(timestamp)
        stats_item.hit()
        self._add_item(stats_item)

    def record_miss(self, cache_item: CacheItem, timestamp=None):
        """
        set hit counter to 0, update last access timestamp
        """
        stats_item = self._stats_item(cache_item)
        timestamp = self._get_timestamp(timestamp)
        stats_item.set_last_access(timestamp)
        stats_item.miss()
        self._add_item(stats_item)

    def record_eviction(self, cache_item: CacheItem, timestamp=None):
        """
        remove item from stats
        """
        stats_item = self._stats_item(cache_item)
        self._remove_item(stats_item)

    def get_active_items(self):
        return self._active_items

    def get_used_capacity(self) -> int:
        """
        currently occupied disk space of this cache in bytes (according to stats, not fs!)
        """
        return sum(i.size for i in self._active_items)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def merge(self, other_stats):
        raise NotImplementedError()

    def serialize(self):
        raise NotImplementedError()

    def save(self, path):
        # TODO: implement me!
        # steps:
        # 1) create and lock a lock file
        # 2) read old cache stats, if they exist
        # 3) merge old and new cache stats
        # 4) write back cache stats
        raise NotImplementedError()

    @classmethod
    def load(self, serialized):
        raise NotImplementedError()
        return CacheStats(...)
