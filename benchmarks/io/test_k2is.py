import os
import glob

import pytest
import numpy as np

from libertem import api
from libertem.udf.masks import ApplyMasksUDF

from utils import drop_cache, warmup_cache, get_testdata_prefixes, backends_by_name


K2IS_FILE = "K2IS/Capture52/Capture52_.gtg"

PREFIXES = get_testdata_prefixes()


def filelist(gtg_file):
    root, ext = os.path.splitext(gtg_file)
    return glob.glob(root + "*")


@pytest.mark.benchmark(
    group="io"
)
@pytest.mark.parametrize(
    "drop", ("cold_cache", "warm_cache")
)
@pytest.mark.parametrize(
    "prefix", PREFIXES
)
def test_sequential(benchmark, prefix, drop):
    hdr = os.path.join(prefix, K2IS_FILE)

    flist = filelist(hdr)

    if drop == "cold_cache":
        drop_cache(flist)
    elif drop == "warm_cache":
        warmup_cache(flist)
    else:
        raise ValueError("bad param")

    benchmark.pedantic(
        warmup_cache, args=(flist, ),
        warmup_rounds=0,
        rounds=1,
        iterations=1
    )

# Starting fresh distributed executors takes a lot of time and therefore
# they should be used repeatedly if possible.
# However, some benchmarks require a fresh distributed executor
# and running several Dask executors in parallel leads to lockups when closing.
# That means any shared executor has to be shut down before a fresh one is started.
# For that reason we use a fixture with scope "class" and group
# tests in a class that should all use the same executor.
# That way we make sure the shared executor is torn down before any other test
# starts a new one.


@pytest.fixture(scope="class")
def shared_dist_ctx():
    print("start shared Context()")
    ctx = api.Context()
    yield ctx
    print("stop shared Context()")
    ctx.close()


class TestUseSharedExecutor:
    @pytest.mark.benchmark(
        group="io"
    )
    @pytest.mark.parametrize(
        "drop", ("cold_cache", "warm_cache")
    )
    @pytest.mark.parametrize(
        "prefix", PREFIXES
    )
    @pytest.mark.parametrize(
        "io_backend", ("mmap", "mmap_readahead", "buffered"),
    )
    def test_mask(self, benchmark, prefix, drop, shared_dist_ctx, io_backend):
        io_backend = backends_by_name[io_backend]
        hdr = os.path.join(prefix, K2IS_FILE)
        flist = filelist(hdr)

        ctx = shared_dist_ctx
        ds = ctx.load(filetype="auto", path=hdr, io_backend=io_backend)

        def mask():
            return np.ones(ds.shape.sig, dtype=bool)

        udf = ApplyMasksUDF(mask_factories=[mask], backends=('numpy', ))

        # warmup executor
        ctx.run_udf(udf=udf, dataset=ds)

        if drop == "cold_cache":
            drop_cache(flist)
        elif drop == "warm_cache":
            warmup_cache(flist)
        else:
            raise ValueError("bad param")

        benchmark.pedantic(
            ctx.run_udf, kwargs=dict(udf=udf, dataset=ds),
            warmup_rounds=0,
            rounds=1,
            iterations=1,
        )


@pytest.mark.benchmark(
    group="io"
)
@pytest.mark.parametrize(
    "first", ("warm_executor", "cold_executor", )
)
@pytest.mark.parametrize(
    "prefix", PREFIXES[:1]
)
@pytest.mark.parametrize(
    "io_backend", ("mmap", "mmap_readahead", "buffered"),
)
def test_mask_firstrun(benchmark, prefix, first, io_backend):
    io_backend = backends_by_name[io_backend]
    hdr = os.path.join(prefix, K2IS_FILE)
    flist = filelist(hdr)

    with api.Context() as ctx:
        ds = ctx.load(filetype="auto", path=hdr, io_backend=io_backend)

        def mask():
            return np.ones(ds.shape.sig, dtype=bool)

        udf = ApplyMasksUDF(mask_factories=[mask], backends=('numpy', ))

        if first == "warm_executor":
            ctx.run_udf(udf=udf, dataset=ds)
        elif first == "cold_executor":
            pass
        else:
            raise ValueError("bad param")

        warmup_cache(flist)

        benchmark.pedantic(
            ctx.run_udf, kwargs=dict(udf=udf, dataset=ds),
            warmup_rounds=0,
            rounds=1,
            iterations=1
        )
