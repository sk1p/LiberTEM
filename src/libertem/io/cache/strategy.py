from .stats import CacheStats, CacheItem


class CacheStrategy:
    def get_victim_list(self, cache_key: str, size: int, stats: CacheStats):
        """
        Return a list of `CacheItem`s that should be deleted to make a new item
        with size in bytes `size`.
        """
        raise NotImplementedError()


class LRUCacheStrategy(CacheStrategy):
    def __init__(self, capacity: int):
        self._capacity = capacity
        super().__init__()

    def get_victim_list(self, cache_key: str, size: int, stats: CacheStats):
        """
        Return a list of `CacheItem`s that should be deleted to make
        place for `partition`.
        """
        # LRU with the following modifications:
        # 1) Don't evict from the same dataset, as our accesses
        #    are highly correlated in a single dataset
        # 2) Include orphaned files as preferred victims
        # 3) TODO: work in an estimated miss cost (challenge: estimate I/O cost
        #    independently from whatever calculation the user decides to run!)
        if self.sufficient_space_for(size, stats):
            return []
        victims = []
        space_to_free = size - self.get_available(stats)

        orphans = stats.get_orphans()

        candidates = stats.query("""
        SELECT dataset, partition, size, path
        FROM stats
        WHERE dataset != ?
        ORDER BY last_access ASC
        """, [cache_key])

        to_check = orphans + [
            CacheItem.from_row(row)
            for row in candidates
        ]

        for item in to_check:
            if space_to_free <= 0:
                break
            victims.append(item)
            space_to_free -= item.size
        if space_to_free > 0:
            raise RuntimeError(
                "not enough cache capacity for the requested operation"
            )  # FIXME: exception class
        return victims

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
