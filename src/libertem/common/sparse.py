import numpy as np

from .array_formats import array_format, as_format, NUMPY, SPARSE_COO, SPARSEFORMATS


def to_dense(a):
    res = as_format(a, NUMPY)
    if res.flags.c_contiguous:
        return res
    else:
        return np.array(res)


def to_sparse(a):
    return as_format(a, SPARSE_COO)


def is_sparse(a):
    return array_format(a) in SPARSEFORMATS
