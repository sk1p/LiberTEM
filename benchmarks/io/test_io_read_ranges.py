import numpy as np
from libertem.common import Shape
from libertem.io.dataset.base.tiling import (
    default_get_read_ranges, TilingScheme
)


def test_read_ranges_many_files(default_raw, benchmark):
    sig_shape = tuple(default_raw.shape.sig)
    tileshape = Shape(
        (int(sig_shape[0] // 2), sig_shape[1]),
        sig_dims=2
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_raw.shape,
    )

    slices_arr = tiling_scheme.slices_array

    num_files = 256*256
    frames_per_file = 4

    fileset_arr = np.zeros((num_files, 4), dtype=np.int64)
    start_idxs = range(0, frames_per_file * num_files, frames_per_file)
    stop_idxs = range(frames_per_file, frames_per_file * num_files + 1, frames_per_file)
    for idx, (start, stop) in enumerate(zip(start_idxs, stop_idxs)):
        fileset_arr[idx] = (start, stop, idx, 7)

    benchmark(
        default_get_read_ranges,
        start_at_frame=1,
        stop_before_frame=frames_per_file * num_files - 17,
        roi=None,
        depth=3,
        slices_arr=slices_arr,
        fileset_arr=fileset_arr,
        sig_shape=tuple(default_raw.shape.sig),
        bpp=8,
        extra=None,
        frame_header_bytes=42,
        frame_footer_bytes=21,
    )
