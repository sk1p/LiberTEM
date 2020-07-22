import functools

import numpy as np
import pytest
from numba.typed import List
from numba.experimental import jitclass
import numba

from libertem.io.dataset.base.backend import _make_mmap_reader_and_decoder
from libertem.io.dataset.base.decode import default_decode
from libertem.io.dataset.base.backend import _decode_as_param, _decode_as_cls, r_n_d_cls

from libertem.io.dataset.k2is import decode_k2is


def default_decode_np(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    out[idx, :] = inp.view(native_dtype)


r_n_d = _make_mmap_reader_and_decoder(
    default_decode,
    enable_jit=True,
    name="decode_jit",
)

r_n_d_nojit = _make_mmap_reader_and_decoder(
    default_decode,
    enable_jit=False,
    name="decode_nojit",
)

r_n_d_nojit_np = _make_mmap_reader_and_decoder(
    default_decode_np,
    enable_jit=False,
    name="decode_nojit_numpy",
)

r_n_d_alt = functools.partial(_decode_as_param, decode=default_decode)
r_n_d_alt.__name__ = "r_n_d_as_param"


@pytest.mark.parametrize("r_n_d", [
    r_n_d, r_n_d_nojit, r_n_d_nojit_np, r_n_d_alt, r_n_d_cls,
])
def test_read_and_decode(benchmark, r_n_d):
    inp = np.random.randn(8, 930, 16).astype(np.float32)
    out = np.zeros_like(inp, dtype=np.float64).reshape((8, 930*16))
    rr = np.array([
        [0, 0*930*16*4, 1*930*16*4],
        [0, 1*930*16*4, 2*930*16*4],
        [0, 2*930*16*4, 3*930*16*4],
        [0, 3*930*16*4, 4*930*16*4],
        [0, 4*930*16*4, 5*930*16*4],
        [0, 5*930*16*4, 6*930*16*4],
        [0, 6*930*16*4, 7*930*16*4],
        [0, 7*930*16*4, 8*930*16*4],
    ])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 930, 16])
    ds_shape = np.array([256, 930, 16])

    mmaps = List()
    mmaps.append(
        inp.reshape((-1,)).view(dtype=np.uint8),
    )

    r_n_d(
        outer_idx=0,
        mmaps=mmaps,
        sig_dims=2,
        tile_read_ranges=rr,
        out_decoded=out,
        native_dtype=np.float32,
        do_zero=False,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )

    benchmark(
        r_n_d,
        outer_idx=0,
        mmaps=mmaps,
        sig_dims=2,
        tile_read_ranges=rr,
        out_decoded=out,
        native_dtype=np.float32,
        do_zero=False,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )


r_n_d_k2is = _make_mmap_reader_and_decoder(
    decode_k2is,
    enable_jit=True,
    name="decode_k2is",
)


r_n_d_k2is_nojit = _make_mmap_reader_and_decoder(
    decode_k2is,
    enable_jit=False,
    name="decode_k2is_nojit",
)


@pytest.mark.parametrize("r_n_d", [
    r_n_d_k2is, r_n_d_k2is_nojit,
])
def test_read_decode_k2is_1(benchmark, r_n_d):
    BLOCK_SHAPE = (930, 16)
    HEADER_SIZE = 40
    BLOCK_SIZE = 0x5758
    DATA_SIZE = BLOCK_SIZE - HEADER_SIZE
    inp = np.random.randn(8*BLOCK_SIZE).astype(np.uint8)
    out = np.zeros((8,) + BLOCK_SHAPE, dtype=np.float64).reshape((8, 930*16))

    rr = np.array([
        [0, 0*BLOCK_SIZE, 1*DATA_SIZE, 1, 1, 0, 0],
        [0, 1*BLOCK_SIZE, 2*DATA_SIZE, 1, 1, 0, 0],
        [0, 2*BLOCK_SIZE, 3*DATA_SIZE, 1, 1, 0, 0],
        [0, 3*BLOCK_SIZE, 4*DATA_SIZE, 1, 1, 0, 0],
        [0, 4*BLOCK_SIZE, 5*DATA_SIZE, 1, 1, 0, 0],
        [0, 5*BLOCK_SIZE, 6*DATA_SIZE, 1, 1, 0, 0],
        [0, 6*BLOCK_SIZE, 7*DATA_SIZE, 1, 1, 0, 0],
        [0, 7*BLOCK_SIZE, 8*DATA_SIZE, 1, 1, 0, 0],
    ])
    origin = np.array([0, 0, 0])
    shape = np.array([8, 930, 16])
    ds_shape = np.array([256, 930, 16])

    mmaps = List()
    mmaps.append(
        inp.reshape((-1,)).view(dtype=np.uint8),
    )

    r_n_d(
        outer_idx=0,
        mmaps=mmaps,
        sig_dims=2,
        tile_read_ranges=rr,
        out_decoded=out,
        native_dtype=np.float32,
        do_zero=False,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )

    benchmark(
        r_n_d,
        outer_idx=0,
        mmaps=mmaps,
        sig_dims=2,
        tile_read_ranges=rr,
        out_decoded=out,
        native_dtype=np.float32,
        do_zero=False,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )


@pytest.mark.parametrize("r_n_d", [
    r_n_d_k2is, r_n_d_k2is_nojit,
])
def test_read_decode_k2is_2(benchmark, r_n_d):
    BLOCK_SHAPE = (930, 16)
    HEADER_SIZE = 40
    BLOCK_SIZE = 0x5758
    DATA_SIZE = BLOCK_SIZE - HEADER_SIZE
    inp = np.random.randn(8*BLOCK_SIZE).astype(np.uint8)
    out = np.zeros((8,) + BLOCK_SHAPE, dtype=np.float64).reshape((4, 930*32))

    rr = np.array([
        [0, 0*BLOCK_SIZE, 1*DATA_SIZE, 1, 2, 0, 0],
        [0, 1*BLOCK_SIZE, 2*DATA_SIZE, 1, 2, 0, 1],
        [0, 2*BLOCK_SIZE, 3*DATA_SIZE, 1, 2, 0, 0],
        [0, 3*BLOCK_SIZE, 4*DATA_SIZE, 1, 2, 0, 1],
        [0, 4*BLOCK_SIZE, 5*DATA_SIZE, 1, 2, 0, 0],
        [0, 5*BLOCK_SIZE, 6*DATA_SIZE, 1, 2, 0, 1],
        [0, 6*BLOCK_SIZE, 7*DATA_SIZE, 1, 2, 0, 0],
        [0, 7*BLOCK_SIZE, 8*DATA_SIZE, 1, 2, 0, 1],
    ])
    origin = np.array([0, 0, 0])
    shape = np.array([4, 930, 32])
    ds_shape = np.array([256, 930, 64])

    mmaps = List()
    mmaps.append(
        inp.reshape((-1,)).view(dtype=np.uint8),
    )

    r_n_d(
        outer_idx=0,
        mmaps=mmaps,
        sig_dims=2,
        tile_read_ranges=rr,
        out_decoded=out,
        native_dtype=np.float32,
        do_zero=False,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )

    benchmark(
        r_n_d,
        outer_idx=0,
        mmaps=mmaps,
        sig_dims=2,
        tile_read_ranges=rr,
        out_decoded=out,
        native_dtype=np.float32,
        do_zero=False,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )
