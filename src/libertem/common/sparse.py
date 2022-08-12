import numpy as np
import scipy.sparse as sp
import sparse


def to_dense(a):
    if isinstance(a, sparse.SparseArray):
        return a.todense()
    elif sp.issparse(a):
        return a.toarray()
    else:
        return np.array(a)


def to_sparse(a):
    if isinstance(a, sparse.COO):
        return a
    elif isinstance(a, sparse.SparseArray):
        return sparse.COO(a)
    elif sp.issparse(a):
        return sparse.COO.from_scipy_sparse(a)
    else:
        return sparse.COO.from_numpy(np.array(a))


def is_sparse(a):
    return isinstance(a, sparse.SparseArray) or sp.issparse(a)


NUMPY = 'numpy.ndarray'
SPARSE_COO = 'sparse.COO'
SPARSE_GCXS = 'sparse.GCXS'

FORMATS = {NUMPY, SPARSE_COO, SPARSE_GCXS}

converters = {
}

for format in FORMATS:
    converters[(format, format)] = lambda x: x
converters[(NUMPY, SPARSE_COO)] = sparse.COO
converters[(NUMPY, SPARSE_GCXS)] = sparse.GCXS
converters[(SPARSE_COO, SPARSE_GCXS)] = sparse.GCXS
converters[(SPARSE_GCXS, SPARSE_COO)] = sparse.COO
converters[(SPARSE_COO, NUMPY)] = to_dense
converters[(SPARSE_GCXS, NUMPY)] = to_dense

classes = {
    NUMPY: np.ndarray,
    SPARSE_COO: sparse.COO,
    SPARSE_GCXS: sparse.GCXS,
}


def array_format(arr):
    for format in FORMATS:
        cls = classes[format]
        if isinstance(arr, cls):
            return format
    return None


def get_converter(source_format, target_format, strict=False):
    identifier = (source_format, target_format)
    if strict:
        return converters[identifier]
    else:
        return converters.get(identifier, lambda x: x)


def as_format(arr, format, strict=True):
    source_format = array_format(arr)
    converter = get_converter(source_format, format, strict)
    return converter(arr)
