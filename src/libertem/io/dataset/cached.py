import os
import json
import hashlib

import numpy as np

from .base import (
    DataSet, Partition, PartitionStructure
)
from libertem.io.dataset.cluster import ClusterDataSet

from libertem.io.cache.stats import CacheItem, CacheStats
from libertem.io.cache.cache import Cache


class CachedDataSet(DataSet):
    """
    Cached DataSet.

    Assumes the source DataSet is significantly slower than the cache location
    (otherwise, it may result in memory pressure, as we don't use direct I/O
    to write to the cache.)

    Parameters
    ----------
    source_ds : DataSet
        DataSet on slower file system

    cache_path : str
        Where should the cache be written to? Should be a directory on a fast
        local device (i.e. NVMe SSD if possible, local hard drive if it is faster than network)
        A subdirectory will be created for each dataset.

    strategy : CacheStrategy
        A class implementing a cache eviction strategy, for example LRUCacheStrategy
    """
    def __init__(self, source_ds, cache_path, strategy, enable_direct=False):
        self._source_ds = source_ds
        self._cache_path = cache_path
        self._cache_key = self._make_cache_key(source_ds.get_cache_key())
        self._path = os.path.join(cache_path, self._cache_key)
        self._enable_direct = enable_direct
        self._cluster_ds = None
        self._executor = None
        self._cache_strategy = strategy

    def _make_cache_key(self, inp):
        inp_as_str = json.dumps(inp)
        return hashlib.sha256(inp_as_str.encode("utf-8")).hexdigest()

    def initialize(self, executor):
        source_structure = PartitionStructure.from_ds(self._source_ds)
        executor.run_each_host(self._ensure_cache_structure)
        cluster_ds = ClusterDataSet(
            path=self._path,
            structure=source_structure,
            enable_direct=self._enable_direct,
        )
        # FIXME: semantics of check_valid: only runs on each host in this case,
        # Context only runs it on a single worker node. I guess Context needs to change?
        executor.run_each_host(cluster_ds.check_valid)
        self._cluster_ds = cluster_ds.initialize(executor=executor)
        self._executor = executor
        return self

    def _get_db_path(self):
        return os.path.join(self._cache_path, "cache.db")

    def _ensure_cache_structure(self):
        os.makedirs(self._path, exist_ok=True)

        cache_stats = CacheStats()
        cache = Cache(stats=cache_stats, strategy=self._cache_strategy)

    @property
    def dtype(self):
        return self._cluster_ds.dtype

    @property
    def shape(self):
        return self._cluster_ds.shape

    @classmethod
    def get_msg_converter(cls):
        raise NotImplementedError(
            "not directly usable from web API"
        )

    def check_valid(self):
        # TODO: validate self._cache_path, what else?
        # - ask cache backend if things look valid (i.e. sidecar cache info is OK)
        return True

    def get_partitions(self):
        for idx, (source_part, cluster_part) in enumerate(zip(self._source_ds.get_partitions(),
                                                              self._cluster_ds.get_partitions())):
            yield CachedPartition(
                source_part=source_part,
                cluster_part=cluster_part,
                meta=cluster_part.meta,
                partition_slice=cluster_part.slice,
                cache_key=self._cache_key,
                cache_strategy=self._cache_strategy,
                db_path=self._get_db_path(),
                idx=idx,
            )

    def evict(self, executor):
        for _ in executor.run_each_partition(self.get_partitions(),
                                             lambda p: p.evict(), all_nodes=True):
            pass

    def __repr__(self):
        return "<CachedDataSet dtype=%s shape=%s source_ds=%s cache_path=%s path=%s>" % (
            self.dtype, self.shape, self._source_ds, self._cache_path, self._path
        )


class CachedPartition(Partition):
    def __init__(self, source_part, cluster_part, meta, partition_slice,
                 cache_key, cache_strategy, db_path, idx):
        super().__init__(meta=meta, partition_slice=partition_slice)
        self._source_part = source_part
        self._cluster_part = cluster_part
        self._cache_key = cache_key
        self._cache_strategy = cache_strategy
        self._db_path = db_path
        self._idx = idx

    def _get_cache(self):
        cache_stats = CacheStats()
        return Cache(stats=cache_stats, strategy=self._cache_strategy)

    def _sizeof(self):
        return self.slice.shape.size * np.dtype(self.dtype).itemsize

    def _write_tiles_noroi(self, wh, source_tiles, dest_dtype):
        """
        Write tiles from source_tiles to the cache. After each tile is written, yield
        it for further processing, potentially doing dtype conversion on the fly.
        """
        with wh:
            miss_tiles = wh.write_tiles(source_tiles)
            if np.dtype(dest_dtype) != np.dtype(self._cluster_part.dtype):
                for tile in miss_tiles:
                    yield tile.astype(dest_dtype)
            else:
                yield from miss_tiles

    def _write_tiles_roi(self, wh, source_tiles, cached_tiles):
        """
        Get source tiles without roi, read and cache whole partition, then
        read all tiles selected via roi from the cache (_cluster_part aka cached_tiles).
        """
        with wh:
            for tile in wh.write_tiles(source_tiles):
                pass
        yield from cached_tiles

    def get_tiles(self, crop_to=None, full_frames=False, mmap=False,
                  dest_dtype="float32", roi=None, target_size=None):
        self._sidechannel
        sc_data = self.sidechannel['dataset']

        cache = self._get_cache()
        cached_tiles = self._cluster_part.get_tiles(crop_to=crop_to, full_frames=full_frames,
                                                   mmap=mmap, dest_dtype=dest_dtype, roi=roi,
                                                   target_size=target_size)
        cache_item = CacheItem(
            dataset=self._cache_key,
            partition=self._idx,
            path=self._cluster_part.get_canonical_path(),
            size=self._sizeof(),
        )
        if self._cluster_part._have_data():
            yield from cached_tiles
            cache.record_hit(cache_item)
        else:
            cache.evict(cache_key=self._cache_key, size=self._sizeof())
            # NOTE: source_tiles are in native dtype!
            source_tiles = self._source_part.get_tiles(
                crop_to=crop_to, full_frames=full_frames, mmap=mmap,
                dest_dtype=self._cluster_part.dtype, roi=None, target_size=target_size
            )
            wh = self._cluster_part.get_write_handle()
            if roi is None:
                yield from self._write_tiles_noroi(wh, source_tiles, dest_dtype)
            else:
                yield from self._write_tiles_roi(wh, source_tiles, cached_tiles)
            cache.record_miss(cache_item)

    def get_locations(self):
        """
        returns locations where this partition is cached
        """
        return self._cluster_part.get_locations()

    def evict(self):
        self._cluster_part.delete()
