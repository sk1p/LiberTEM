import numpy as np

from libertem.common import Shape


class UDFMeta:
    """
    UDF metadata. Makes all relevant metadata accessible to the UDF. Can be different
    for each task/partition.

    .. versionchanged:: 0.4.0.dev0
        Added distinction of dataset_dtype and input_dtype
    """
    def __init__(self, partition_shape, dataset_shape, roi, dataset_dtype, input_dtype):
        self._partition_shape = partition_shape
        self._dataset_shape = dataset_shape
        self._dataset_dtype = dataset_dtype
        self._input_dtype = input_dtype
        if roi is not None:
            roi = roi.reshape(dataset_shape.nav)
        self._roi = roi
        self._slice = None

    @property
    def slice(self):
        """
        Slice : A :class:`~libertem.common.slice.Slice` instance that describes the location
                within the dataset with navigation dimension flattened and reduced to the ROI.
        """
        return self._slice

    @slice.setter
    def slice(self, new_slice):
        self._slice = new_slice

    @property
    def partition_shape(self) -> Shape:
        """
        Shape : The shape of the partition this UDF currently works on.
                If a ROI was applied, the shape will be modified accordingly.
        """
        return self._partition_shape

    @property
    def dataset_shape(self) -> Shape:
        """
        Shape : The original shape of the whole dataset, not influenced by the ROI
        """
        return self._dataset_shape

    @property
    def roi(self) -> np.ndarray:
        """
        numpy.ndarray : Boolean array which limits the elements the UDF is working on.
                     Has a shape of :attr:`dataset_shape.nav`.
        """
        return self._roi

    @property
    def dataset_dtype(self) -> np.dtype:
        """
        numpy.dtype : Native dtype of the dataset
        """
        return self._dataset_dtype

    @property
    def input_dtype(self) -> np.dtype:
        """
        numpy.dtype : dtype of the data that will be passed to the UDF

        This is determined from the dataset's native dtype and
        :meth:`UDF.get_preferred_input_dtype` using :meth:`numpy.result_type`

        .. versionadded:: 0.4.0.dev0
        """
        return self._input_dtype
