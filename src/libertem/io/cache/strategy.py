from .stats import CacheStats, CacheItem
from .base import CacheException


class CacheStrategy:
    def get_victim_list(self, cache_key: str, size: int, stats: CacheStats):
        """
        Return a list of `CacheItem`s that are candidates for deletion to make
        place for a new item with size in bytes `size`.
        """
        raise NotImplementedError()


class LRUCacheStrategy(CacheStrategy):
    def __init__(self, capacity: int):
        """
        Parameters
        ----------

        capacity
            Size of the cache, in bytes
        """
        self._capacity = capacity
        super().__init__()

    def get_victim_list(self, cache_key: str, size: int, stats: CacheStats):
        """
        Return a list of `CacheItem`s that are candidates for deletion to make
        place for a new item with size in bytes `size`.

        Parameters
        ----------

        cache_key
            The cache key of the DataSet for which we
            need to make place.

        size
            How much space is needed? in bytes
        """
        # LRU with the following modifications:
        # 1) Don't evict from the same dataset, as our accesses
        #    are highly correlated in a single dataset
        # 2) TODO: Include orphaned files as preferred victims
        # 3) TODO: work in an estimated miss cost (challenge: estimate I/O cost
        #    independently from whatever calculation the user decides to run!)
        if self.sufficient_space_for(size, stats):
            return []
        victims = []
        space_to_free = size - self.get_available(stats)

        active_items = stats.get_active_items()

        candidates = sorted([
            item
            for item in active_items 
            if item.dataset != cache_key
        ], key=lambda item: item.last_access)

        if sum(i.size for i in candidates) < space_to_free:
            raise CacheException(
                "not enough cache capacity for the requested operation"
            )

        return candidates

    def sufficient_space_for(self, size: int, stats: CacheStats):
        return size <= self.get_available(stats)

    def get_available(self, stats: CacheStats):
        """
        available cache capacity in bytes
        """
        return self._capacity - self.get_used(stats)

    def get_used(self, stats: CacheStats):
        """
        used cache capacity in bytes
        """
        return stats.get_used_capacity()
