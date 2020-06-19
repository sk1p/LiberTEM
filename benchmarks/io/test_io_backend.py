import functools
import numpy as np
import pytest
from numba.typed import List

from libertem.io.dataset.base.backend import _make_mmap_reader_and_decoder
from libertem.io.dataset.base.decode import default_decode
from libertem.io.dataset.base.backend import _decode_as_param


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


@pytest.mark.parametrize("r_n_d", [
    r_n_d, r_n_d_nojit, r_n_d_nojit_np, r_n_d_alt,
])
def test_read_and_decode(benchmark, r_n_d):
    inp = np.random.randn(8, 16, 16).astype(np.float32)
    out = np.zeros_like(inp, dtype=np.float64).reshape((8, 256))
    rr = np.array([
        [0, 0*16*16*4, 1*16*16*4],
        [0, 1*16*16*4, 2*16*16*4],
        [0, 2*16*16*4, 3*16*16*4],
        [0, 3*16*16*4, 4*16*16*4],
        [0, 4*16*16*4, 5*16*16*4],
        [0, 5*16*16*4, 6*16*16*4],
        [0, 6*16*16*4, 7*16*16*4],
        [0, 7*16*16*4, 8*16*16*4],
    ])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 16, 16])
    ds_shape = np.array([256, 16, 16])

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
