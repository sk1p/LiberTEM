import numpy as np
import pytest

from libertem.common import Shape
from libertem.io.dataset.base.tiling import (
    default_get_read_ranges, TilingScheme
)


@pytest.mark.with_numba
def test_default_read_ranges():
    sig_shape = (128, 128)
    dataset_shape = Shape(
        (16, 16, 128, 128),
        sig_dims=2
    )
    tileshape = Shape(
        (int(sig_shape[0] // 2), sig_shape[1]),
        sig_dims=2
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=dataset_shape,
    )

    slices_arr = tiling_scheme.slices_array
    frame_header_bytes = 42
    frame_footer_bytes = 21
    bpp = 8
    depth = 4
    start_frame = 3
    stop_before_frame = 7
    num_files = 4

    fileset_arr = np.zeros((num_files, 4), dtype=np.int64)
    #                (start, stop, fileset index, file_header_bytes)
    fileset_arr[0] = (0,    4, 0, 7)
    fileset_arr[1] = (4,    8, 1, 3)
    fileset_arr[2] = (8,   12, 2, 9)
    fileset_arr[3] = (12, 256, 3, 2)

    tile_slices, read_ranges, scheme_indices = default_get_read_ranges(
        start_at_frame=start_frame,
        stop_before_frame=stop_before_frame,
        roi=None,
        depth=depth,
        slices_arr=slices_arr,
        fileset_arr=fileset_arr,
        sig_shape=sig_shape,
        bpp=bpp,
        extra=None,
        frame_header_bytes=frame_header_bytes,
        frame_footer_bytes=frame_footer_bytes,
    )

    # result is two tiles, for the two slices in the tiling scheme:
    assert tile_slices.shape == (2, 2, 3)
    assert read_ranges.shape == (2, depth, 3)
    assert scheme_indices.shape == (2,)
    assert np.allclose(scheme_indices, (0, 1))

    # first tile:
    assert np.allclose(tile_slices[0][0], (3, 0, 0))  # origin
    assert np.allclose(tile_slices[0][1], (depth, 64, 128))  # shape
    assert np.allclose(
        read_ranges[0],
        [
            # snapshot of known-good values:
            # (file_index, start_bytes, end_bytes)
            [0, 393447, 458983],
            [1,     42,  65578],
            [1, 131177, 196713],
            [1, 262312, 327848]
        ],
    )

    # second tile:
    assert np.allclose(tile_slices[1][0], (3, 64, 0))  # origin
    assert np.allclose(tile_slices[1][1], (depth, 64, 128))  # shape
    assert np.allclose(
        read_ranges[1],
        [
            # snapshot of known-good values:
            # (file_index, start_bytes, end_bytes)
            [0, 458983, 524519],
            [1,  65578, 131114],
            [1, 196713, 262249],
            [1, 327848, 393384]
        ],
    )


@pytest.mark.with_numba
def test_default_read_ranges_with_roi():
    sig_shape = (128, 128)
    dataset_shape = Shape(
        (16, 16, 128, 128),
        sig_dims=2
    )
    tileshape = Shape(
        (int(sig_shape[0] // 2), sig_shape[1]),
        sig_dims=2
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=dataset_shape,
    )

    roi = np.zeros(np.prod(dataset_shape.nav), dtype=bool)
    roi[::2] = True  # select every second frame

    slices_arr = tiling_scheme.slices_array
    frame_header_bytes = 42
    frame_footer_bytes = 21
    bpp = 8
    depth = 4
    start_frame = 3
    stop_before_frame = 7
    num_files = 4

    fileset_arr = np.zeros((num_files, 4), dtype=np.int64)
    #                (start, stop, fileset index, file_header_bytes)
    fileset_arr[0] = (0, 4,  0, 7)
    fileset_arr[1] = (4, 8,  1, 3)
    fileset_arr[2] = (8,   12, 2, 9)
    fileset_arr[3] = (12, 256, 3, 2)

    tile_slices, read_ranges, scheme_indices = default_get_read_ranges(
        start_at_frame=start_frame,
        stop_before_frame=stop_before_frame,
        roi=roi,
        depth=depth,
        slices_arr=slices_arr,
        fileset_arr=fileset_arr,
        sig_shape=sig_shape,
        bpp=bpp,
        extra=None,
        frame_header_bytes=frame_header_bytes,
        frame_footer_bytes=frame_footer_bytes,
    )

    # result is two tiles, for the two slices in the tiling scheme:
    assert tile_slices.shape == (2, 2, 3)
    # depth // 2 because we only select every other frame in the given (start, stop) range
    assert read_ranges.shape == (2, depth // 2, 3)
    assert scheme_indices.shape == (2,)
    assert np.allclose(scheme_indices, (0, 1))

    # NOTE: compare with the test case above without roi,
    # the roi selects every second line in the read ranges

    # first tile:
    assert np.allclose(tile_slices[0][0], (2, 0, 0))  # origin (roi-compressed)
    assert np.allclose(tile_slices[0][1], (depth // 2, 64, 128))  # shape
    assert np.allclose(
        read_ranges[0],
        [
            # snapshot of known-good values:
            # (file_index, start_bytes, end_bytes)
            [1,     42,  65578],
            [1, 262312, 327848]
        ],
    )

    # second tile:
    assert np.allclose(tile_slices[1][0], (2, 64, 0))  # origin (roi-compressed)
    assert np.allclose(tile_slices[1][1], (depth // 2, 64, 128))  # shape
    assert np.allclose(
        read_ranges[1],
        [
            # snapshot of known-good values:
            # (file_index, start_bytes, end_bytes)
            [1,  65578, 131114],
            [1, 327848, 393384]
        ],
    )
