import typing

import numba
import numpy as np
import pytest

from libertem.io.dataset.base.decode import (
    decode_swap_2, decode_swap_4, decode_swap_8,
    DtypeConversionDecoder,
)


@numba.njit(inline='always', cache=True)
def dd_1(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    # out[idx, :] = inp.view(np.uint16)
    out[idx, :] = inp.view(native_dtype)


@numba.njit(inline='always', cache=True)
def dd_2(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    # out[idx, :] = inp.view(np.uint16)
    out[idx, :] = inp.view(native_dtype)


@pytest.mark.parametrize("fn", [dd_1, dd_2])
@pytest.mark.with_numba
def test_default_decode(benchmark, fn):
    inp = np.zeros((1, 16, 16)).astype(np.uint16)
    out = np.zeros_like(inp, dtype=np.float32)
    rr = np.array([0, 0, inp.nbytes])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 16, 16])
    ds_shape = np.array([256, 16, 16])
    benchmark(
        fn,
        inp.reshape((-1,)).view(dtype=np.uint8),
        out=out.reshape((1, -1,)),
        idx=0,
        native_dtype=inp.dtype,
        rr=rr,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )


@pytest.mark.with_numba
def test_decode_swap_2_u16(benchmark):
    inp = np.random.randint(low=0, high=2**15, size=(1, 16, 16), dtype=np.uint16)
    out = np.zeros_like(inp, dtype=inp.dtype)
    rr = np.array([0, 0, inp.nbytes])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 16, 16])
    ds_shape = np.array([256, 16, 16])
    inp_swapped = inp.reshape((-1,)).byteswap()
    benchmark(
        decode_swap_2,
        inp=inp_swapped.view(dtype=np.uint8),
        out=out.reshape((1, -1)),
        idx=0,
        native_dtype=inp.dtype,
        rr=rr,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )


@pytest.mark.with_numba
def test_decode_swap_4_u32(benchmark):
    inp = np.random.randint(low=0, high=2**31, size=(1, 16, 16), dtype=np.uint32)
    out = np.zeros_like(inp, dtype=inp.dtype)
    rr = np.array([0, 0, inp.nbytes])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 16, 16])
    ds_shape = np.array([256, 16, 16])
    inp_swapped = inp.reshape((-1,)).byteswap()
    benchmark(
        decode_swap_4,
        inp=inp_swapped.view(dtype=np.uint8),
        out=out.reshape((1, -1)),
        idx=0,
        native_dtype=inp.dtype,
        rr=rr,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )


@pytest.mark.with_numba
def test_decode_swap_4_u64(benchmark):
    inp = np.random.randint(low=0, high=2**31, size=(1, 16, 16), dtype=np.uint64)
    out = np.zeros_like(inp, dtype=inp.dtype)
    rr = np.array([0, 0, inp.nbytes])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 16, 16])
    ds_shape = np.array([256, 16, 16])
    inp_swapped = inp.reshape((-1,)).byteswap()
    benchmark(
        decode_swap_8,
        inp=inp_swapped.view(dtype=np.uint8),
        out=out.reshape((1, -1)),
        idx=0,
        native_dtype=inp.dtype,
        rr=rr,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )


def make_decode_params():
    in_bos = ['<', '>', '=']  # input byteorder
    o_bos = ['=']  # decoding to non-native byteorder is not supported currently
    dtypes = [
        ("u1->u2", np.uint8, np.uint16),
        ("u1->u4", np.uint8, np.uint32),
        ("u1->u8", np.uint8, np.uint64),
        ("u2->u2", np.uint16, np.uint16),
        ("u2->u4", np.uint16, np.uint32),
        ("u2->u8", np.uint16, np.uint64),
        ("u4->u4", np.uint32, np.uint32),
        ("u4->u8", np.uint32, np.uint64),

        ("u1->f4", np.uint8, np.float32),
        ("u2->f4", np.uint16, np.float32),
        ("u4->f4", np.uint32, np.float32),

        ("u1->f8", np.uint8, np.float64),
        ("u2->f8", np.uint16, np.float64),
        ("u4->f8", np.uint32, np.float64),
    ]

    return [
        (name, (d1, d2), in_bo, o_bo)
        for name, d1, d2 in dtypes
        for in_bo in in_bos
        for o_bo in o_bos
    ]


@pytest.mark.parametrize(
    'dtype_names, dtypes, in_byteorder, out_byteorder',
    make_decode_params()
)
@pytest.mark.with_numba
def test_default_decoder_from_uint(
    dtype_names: str,
    dtypes: typing.Tuple[np.dtype],
    in_byteorder, out_byteorder, benchmark,
):
    in_dtype, out_dtype = dtypes
    decoder = DtypeConversionDecoder()
    in_dtype = np.dtype(in_dtype)
    out_dtype = np.dtype(out_dtype)
    in_dtype_full = in_dtype.newbyteorder(in_byteorder)
    out_dtype_full = out_dtype.newbyteorder(out_byteorder)
    decode = decoder.get_decode(native_dtype=in_dtype_full, read_dtype=out_dtype_full)
    inp = np.random.randint(
        low=0,
        high=np.iinfo(in_dtype).max,
        size=(1, 16, 16),
        dtype=in_dtype
    )
    if in_dtype_full != in_dtype:
        inp.byteswap()
    print(in_dtype_full, in_dtype, out_dtype_full, out_dtype)
    out = np.zeros_like(inp, dtype=out_dtype_full)
    rr = np.array([0, 0, inp.nbytes])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 16, 16])
    ds_shape = np.array([256, 16, 16])

    benchmark(
        decode,
        inp=inp.reshape((-1,)).view(dtype=np.uint8),
        out=out.reshape((1, -1)),
        idx=0,
        native_dtype=inp.dtype,
        rr=rr,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )
