import os
import pickle
import json

import pytest
import numpy as np

from libertem.io.dataset.frms6 import (
    FRMS6DataSet, _map_y, FRMS6Decoder,
    frms6_get_read_ranges,
)
from libertem.analysis.raw import PickFrameAnalysis
from libertem.analysis.sum import SumAnalysis
from libertem.io.dataset.base import TilingScheme
from libertem.common import Shape
from libertem.udf.raw import PickUDF

from utils import dataset_correction_verification

FRMS6_TESTDATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                 'data', 'frms6', 'C16_15_24_151203_019.hdr')
HAVE_FRMS6_TESTDATA = os.path.exists(FRMS6_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_FRMS6_TESTDATA, reason="need frms6 testdata")  # NOQA


@pytest.fixture
def default_frms6(lt_ctx):
    ds = FRMS6DataSet(path=FRMS6_TESTDATA_PATH)
    ds = ds.initialize(lt_ctx.executor)
    return ds


@pytest.fixture
def dist_frms6(dist_ctx):
    path = "/data/frms6/C16_15_24_151203_019.hdr"
    ds = FRMS6DataSet(path=path)
    ds = ds.initialize(dist_ctx.executor)
    return ds


def test_simple_open(default_frms6):
    assert tuple(default_frms6.shape) == (256, 256, 264, 264)


def test_detetct(lt_ctx):
    assert FRMS6DataSet.detect_params(
        FRMS6_TESTDATA_PATH, lt_ctx.executor
    )["parameters"] is not False


def test_check_valid(default_frms6):
    default_frms6.check_valid()


def test_sum_analysis(default_frms6, lt_ctx):
    roi = {
        "shape": "disk",
        "cx": 5,
        "cy": 6,
        "r": 7,
    }
    analysis = SumAnalysis(dataset=default_frms6, parameters={
        "roi": roi,
    })
    # not checking result yet, just making sure it doesn't crash:
    lt_ctx.run(analysis)


def test_pick_job(default_frms6, lt_ctx):
    analysis = lt_ctx.create_pick_job(dataset=default_frms6, origin=(16, 16))
    results = lt_ctx.run(analysis)
    assert results.shape == (264, 264)


@pytest.mark.parametrize(
    'TYPE', ['JOB', 'UDF']
)
def test_pick_analysis(default_frms6, lt_ctx, TYPE):
    analysis = PickFrameAnalysis(dataset=default_frms6, parameters={"x": 16, "y": 16})
    analysis.TYPE = TYPE
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (264, 264)


@pytest.mark.parametrize(
    # Default is too large for test without ROI
    "with_roi", (True, )
)
def test_correction(default_frms6, lt_ctx, with_roi):
    ds = default_frms6

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


def test_pickle_is_small(default_frms6):
    pickled = pickle.dumps(default_frms6)
    pickle.loads(pickled)

    # because of the dark frame stuff, the dataset is actually quite large:
    assert len(pickled) < 300 * 1024


def test_cache_key_json_serializable(default_frms6):
    json.dumps(default_frms6.get_cache_key())


@pytest.mark.dist
def test_dist_process(dist_frms6, dist_ctx):
    roi = {
        "shape": "disk",
        "cx": 5,
        "cy": 6,
        "r": 7,
    }
    analysis = SumAnalysis(dataset=dist_frms6, parameters={"roi": roi})
    dist_ctx.run(analysis)


@pytest.mark.dist
def test_initialize(dist_frms6, dist_ctx):
    assert dist_frms6._filenames is not None
    assert dist_frms6._hdr_info is not None
    assert dist_frms6._hdr_info is not None

# TODO: gain map tests
# TODO: test load request message
# TODO: test error conditions


def test_map_y():
    assert _map_y(y=0, xs=264, binning=4, num_rows=264) == (0, 0)
    assert _map_y(y=32, xs=264, binning=4, num_rows=264) == (32, 0)
    assert _map_y(y=33, xs=264, binning=4, num_rows=264) == (32, 264)
    assert _map_y(y=65, xs=264, binning=4, num_rows=264) == (0, 264)

    assert _map_y(y=0, xs=264, binning=1, num_rows=264) == (0, 0)
    assert _map_y(y=32, xs=264, binning=1, num_rows=264) == (32, 0)
    assert _map_y(y=33, xs=264, binning=1, num_rows=264) == (33, 0)
    assert _map_y(y=65, xs=264, binning=1, num_rows=264) == (65, 0)
    assert _map_y(y=131, xs=264, binning=1, num_rows=264) == (131, 0)
    assert _map_y(y=132, xs=264, binning=1, num_rows=264) == (131, 264)
    assert _map_y(y=263, xs=264, binning=1, num_rows=264) == (0, 264)


@pytest.mark.with_numba
@pytest.mark.parametrize(
    'binning', [1, 2, 4],
)
def test_decode(binning):
    out = np.zeros((8, 8, 264), dtype=np.uint16)
    reads = [
        np.random.randint(low=1, high=1024, size=(1, 264), dtype=np.uint16)
        for i in range(out.shape[0] * out.shape[1] // binning)
    ]

    decoder = FRMS6Decoder(binning=binning)
    decode = decoder.get_decode(native_dtype="u2", read_dtype=np.float32)

    for idx, read in enumerate(reads):
        decode(
            inp=read,
            out=out,
            idx=idx,
            native_dtype=np.uint16,
            rr=None,
            origin=np.array([0, 0, 0]),
            shape=np.array(out.shape),
            ds_shape=np.array([1024, 264, 264]),
        )

    for idx, px in enumerate(out.reshape((-1,))):
        assert not np.isclose(px, 0)


@pytest.mark.with_numba
def test_with_roi(default_frms6, lt_ctx):
    udf = PickUDF()
    roi = np.zeros(default_frms6.shape.nav, dtype=bool)
    roi[0] = 1
    res = lt_ctx.run_udf(udf=udf, dataset=default_frms6, roi=roi)
    np.array(res['intensity']).shape == (1, 256, 256)


def test_read_invalid_tileshape(default_frms6):
    partitions = default_frms6.get_partitions()
    p = next(partitions)

    tileshape = Shape(
        (1, 3, 264),
        sig_dims=2,
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_frms6.shape,
    )

    with pytest.raises(ValueError):
        next(p.get_tiles(tiling_scheme=tiling_scheme))


@pytest.mark.with_numba
def test_frms6_read_ranges(default_frms6):
    sig_shape = tuple(default_frms6.shape.sig)
    tileshape = Shape(
        (4, int(sig_shape[0] // 2), sig_shape[1]),
        sig_dims=2
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_frms6.shape,
    )

    fileset = default_frms6._get_fileset()

    start_frame = 3
    stop_before_frame = 7

    tile_slices, read_ranges, scheme_indices = fileset.get_read_ranges(
        start_at_frame=start_frame,
        stop_before_frame=stop_before_frame,
        dtype=default_frms6.raw_dtype,
        tiling_scheme=tiling_scheme,
        roi=None,
    )

    # result is two tiles, for the two slices in the tiling scheme:
    assert tile_slices.shape == (2, 2, 3)
    # frms6 has individual reads for each line, keep binning in mind!
    assert read_ranges.shape == (2, 132, 3)
    assert scheme_indices.shape == (2,)
    assert np.allclose(scheme_indices, (0, 1))

    # first tile:
    assert np.allclose(tile_slices[0][0], (3, 0, 0))  # origin
    assert np.allclose(tile_slices[0][1], (tiling_scheme.depth, 132, 264))  # shape
    assert np.allclose(
        # let's check the first 16 values:
        read_ranges[0][:16],
        [
            # snapshot of known-good values:
            # (file_index, start_bytes, end_bytes)
            [0, 104800, 105328],
            [0, 105856, 106384],
            [0, 106912, 107440],
            [0, 107968, 108496],
            [0, 109024, 109552],
            [0, 110080, 110608],
            [0, 111136, 111664],
            [0, 112192, 112720],
            [0, 113248, 113776],
            [0, 114304, 114832],
            [0, 115360, 115888],
            [0, 116416, 116944],
            [0, 117472, 118000],
            [0, 118528, 119056],
            [0, 119584, 120112],
            [0, 120640, 121168]
        ],
    )

    # second tile:
    assert np.allclose(tile_slices[1][0], (3, 132, 0))  # origin
    assert np.allclose(tile_slices[1][1], (tiling_scheme.depth, 132, 264))  # shape
    assert np.allclose(
        # let's check the first 16 values:
        read_ranges[1][:16],
        [
            # snapshot of known-good values:
            # (note that they are in descending order because of the
            # folding of data in frms6)
            # (file_index, start_bytes, end_bytes)
            [0, 139120, 139648],
            [0, 138064, 138592],
            [0, 137008, 137536],
            [0, 135952, 136480],
            [0, 134896, 135424],
            [0, 133840, 134368],
            [0, 132784, 133312],
            [0, 131728, 132256],
            [0, 130672, 131200],
            [0, 129616, 130144],
            [0, 128560, 129088],
            [0, 127504, 128032],
            [0, 126448, 126976],
            [0, 125392, 125920],
            [0, 124336, 124864],
            [0, 123280, 123808]
        ],
    )


def test_frms6_read_ranges_with_roi(default_frms6):
    roi = np.zeros(np.prod(default_frms6.shape.nav), dtype=bool)
    roi[::2] = True  # select every second frame

    sig_shape = tuple(default_frms6.shape.sig)
    tileshape = Shape(
        (4, int(sig_shape[0] // 2), sig_shape[1]),
        sig_dims=2
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_frms6.shape,
    )

    fileset = default_frms6._get_fileset()

    start_frame = 3
    stop_before_frame = 7

    tile_slices, read_ranges, scheme_indices = fileset.get_read_ranges(
        start_at_frame=start_frame,
        stop_before_frame=stop_before_frame,
        dtype=default_frms6.raw_dtype,
        tiling_scheme=tiling_scheme,
        roi=roi,
    )

    # result is two tiles, for the two slices in the tiling scheme:
    assert tile_slices.shape == (2, 2, 3)
    # frms6 has individual reads for each line, keep binning in mind!
    assert read_ranges.shape == (2, 66, 3)
    assert scheme_indices.shape == (2,)
    assert np.allclose(scheme_indices, (0, 1))

    # first tile:
    assert np.allclose(tile_slices[0][0], (2, 0, 0))  # origin
    assert np.allclose(tile_slices[0][1], (tiling_scheme.depth // 2, 132, 264))  # shape
    assert np.allclose(
        # let's check the first 16 values:
        read_ranges[0][:16],
        [
            # snapshot of known-good values:
            # (file_index, start_bytes, end_bytes)
            [0, 139712, 140240],
            [0, 140768, 141296],
            [0, 141824, 142352],
            [0, 142880, 143408],
            [0, 143936, 144464],
            [0, 144992, 145520],
            [0, 146048, 146576],
            [0, 147104, 147632],
            [0, 148160, 148688],
            [0, 149216, 149744],
            [0, 150272, 150800],
            [0, 151328, 151856],
            [0, 152384, 152912],
            [0, 153440, 153968],
            [0, 154496, 155024],
            [0, 155552, 156080]
        ],
    )

    # second tile:
    assert np.allclose(tile_slices[1][0], (2, 132, 0))  # origin
    assert np.allclose(tile_slices[1][1], (tiling_scheme.depth // 2, 132, 264))  # shape
    assert np.allclose(
        # let's check the first 16 values:
        read_ranges[1][:16],
        [
            # snapshot of known-good values:
            # (note that they are in descending order because of the
            # folding of data in frms6)
            # (file_index, start_bytes, end_bytes)
            [0, 174032, 174560],
            [0, 172976, 173504],
            [0, 171920, 172448],
            [0, 170864, 171392],
            [0, 169808, 170336],
            [0, 168752, 169280],
            [0, 167696, 168224],
            [0, 166640, 167168],
            [0, 165584, 166112],
            [0, 164528, 165056],
            [0, 163472, 164000],
            [0, 162416, 162944],
            [0, 161360, 161888],
            [0, 160304, 160832],
            [0, 159248, 159776],
            [0, 158192, 158720]
        ],
    )
