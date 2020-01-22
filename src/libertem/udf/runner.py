import uuid

import cloudpickle
import numpy as np

from libertem.common import Shape, Slice
from .meta import UDFMeta
from libertem.utils import SideChannel


class Task(object):
    """
    A computation on a partition. Inherit from this class and implement ``__call__``
    for your specific computation.

    .. versionchanged:: 0.4.0.dev0
        Moved from libertem.job.base to libertem.udf.base as part of Job API deprecation
    """

    def __init__(self, partition, idx):
        self.partition = partition
        self.idx = idx

    def get_locations(self):
        return self.partition.get_locations()

    def __call__(self):
        raise NotImplementedError()


class UDFTask:
    def __init__(self, partition, idx, udf, roi, sidechannel):
        # FIXME: underscores?
        self.partition = partition
        self.idx = idx
        self._roi = roi
        self._udf = udf
        self._sidechannel = sidechannel

    def get_locations(self):
        return self.partition.get_locations()

    def __call__(self):
        return UDFRunner(self._udf).run_for_partition(self.partition, self._roi, self._sidechannel)


class UDFRunner:
    """
    `UDFRunner` is the main class responsible for execution of UDFs. It takes
    care of allocating buffers, and forwarding all relevant information to the UDF
    instance. It also contains the main loop that reads from the `DataSet` and
    calls the approriate `process_*` function.
    """

    def __init__(self, udf, debug=False):
        self._udf = udf
        self._debug = debug

    def _get_dtype(self, dtype):
        return np.result_type(self._udf.get_preferred_input_dtype(), dtype)

    def run_for_partition(self, partition, roi, sidechannel):
        dtype = self._get_dtype(partition.dtype)
        meta = UDFMeta(
            partition_shape=partition.slice.adjust_for_roi(roi).shape,
            dataset_shape=partition.meta.shape,
            roi=roi,
            dataset_dtype=dtype,
            input_dtype=dtype,
        )
        self._udf.set_meta(meta)
        self._udf.init_result_buffers()
        self._udf.allocate_for_part(partition, roi)
        self._udf.init_task_data()
        if hasattr(self._udf, 'preprocess'):
            self._udf.clear_views()
            self._udf.preprocess()
        method = self._udf.get_method()
        if method == 'tile':
            tiles = partition.get_tiles(full_frames=False, roi=roi, dest_dtype=dtype, mmap=True)
        elif method == 'frame':
            tiles = partition.get_tiles(full_frames=True, roi=roi, dest_dtype=dtype, mmap=True)
        elif method == 'partition':
            tiles = [partition.get_macrotile(roi=roi, dest_dtype=dtype, mmap=True)]

        for tile in tiles:
            if method == 'tile':
                self._udf.set_views_for_tile(partition, tile)
                self._udf.set_slice(tile.tile_slice)
                self._udf.process_tile(tile.data)
            elif method == 'frame':
                tile_slice = tile.tile_slice
                for frame_idx, frame in enumerate(tile.data):
                    frame_slice = Slice(
                        origin=(tile_slice.origin[0] + frame_idx,) + tile_slice.origin[1:],
                        shape=Shape((1,) + tuple(tile_slice.shape)[1:],
                                    sig_dims=tile_slice.shape.sig.dims),
                    )
                    self._udf.set_slice(frame_slice)
                    self._udf.set_views_for_frame(partition, tile, frame_idx)
                    self._udf.process_frame(frame)
            elif method == 'partition':
                self._udf.set_views_for_tile(partition, tile)
                self._udf.set_slice(partition.slice)
                self._udf.process_partition(tile.data)

        if hasattr(self._udf, 'postprocess'):
            self._udf.clear_views()
            self._udf.postprocess()

        self._udf.cleanup()
        self._udf.clear_views()

        if self._debug:
            try:
                cloudpickle.loads(cloudpickle.dumps(partition))
            except TypeError:
                raise TypeError("could not pickle partition")
            try:
                cloudpickle.loads(cloudpickle.dumps(self._udf.results))
            except TypeError:
                raise TypeError("could not pickle results")

        return self._udf.results, partition

    def _prepare_run_for_dataset(self, dataset, executor, roi):
        if roi is not None:
            if np.product(roi.shape) != np.product(dataset.shape.nav):
                raise ValueError(
                    "roi: incompatible shapes: %s (roi) vs %s (dataset)" % (
                        roi.shape, dataset.shape.nav
                    )
                )
        meta = UDFMeta(
            partition_shape=None,
            dataset_shape=dataset.shape,
            roi=roi,
            dataset_dtype=self._get_dtype(dataset.dtype),
            input_dtype=self._get_dtype(dataset.dtype),
        )
        self._udf.set_meta(meta)
        self._udf.init_result_buffers()
        self._udf.allocate_for_full(dataset, roi)

        sidechannel = SideChannel()
        tasks = self._make_udf_tasks(dataset, roi, sidechannel)

        return sidechannel, tasks

    def run_for_dataset(self, dataset, executor, roi=None):
        cancel_id = str(uuid.uuid4())

        sidechannel, tasks = self._prepare_run_for_dataset(dataset, executor, roi)

        dataset.hook_before_udf_run(executor, sidechannel, roi)

        if self._debug:
            tasks = list(tasks)
            cloudpickle.loads(cloudpickle.dumps(tasks))

        for part_results, partition in executor.run_tasks(tasks, cancel_id):
            self._udf.set_views_for_partition(partition)
            self._udf.merge(
                dest=self._udf.results.get_proxy(),
                src=part_results.get_proxy()
            )

        self._udf.clear_views()

        dataset.hook_after_udf_run(executor, sidechannel, roi)

        return self._udf.results.as_dict()

    async def run_for_dataset_async(self, dataset, executor, cancel_id, roi=None):
        sidechannel, tasks = self._prepare_run_for_dataset(dataset, executor, roi)

        # FIXME: convert to sync executor and run in different thread
        dataset.hook_before_udf_run(executor, sidechannel, roi)

        async for part_results, partition in executor.run_tasks(tasks, cancel_id):
            self._udf.set_views_for_partition(partition)
            self._udf.merge(
                dest=self._udf.results.get_proxy(),
                src=part_results.get_proxy()
            )
            self._udf.clear_views()
            yield self._udf.results.as_dict()
        else:
            # yield at least one result (which should be empty):
            self._udf.clear_views()
            yield self._udf.results.as_dict()

        # FIXME: convert to sync executor and run in different thread
        dataset.hook_after_udf_run(executor, sidechannel, roi)

    def _roi_for_partition(self, roi, partition):
        return roi.reshape(-1)[partition.slice.get(nav_only=True)]

    def _make_udf_tasks(self, dataset, roi, sidechannel):
        for idx, partition in enumerate(dataset.get_partitions()):
            if roi is not None:
                roi_for_part = self._roi_for_partition(roi, partition)
                if np.count_nonzero(roi_for_part) == 0:
                    # roi is empty for this partition, ignore
                    continue
            udf = self._udf.copy_for_partition(partition, roi)
            yield UDFTask(partition=partition, idx=idx, udf=udf, roi=roi, sidechannel=sidechannel)
