# -*- coding: utf-8 -*-
"""Benchmark the cell orderings and accuflux on a large flow direction raster.

Input is a flat ``int32`` array of next-downstream indices, one per cell: ``-1``
for nodata, and a cell that drains to itself is a pit. This is the layout
pyflwdir itself uses, so any raster can be exported to it with
``flw.idxs_ds.astype("int32").tofile(path)``.

    python bench_large.py directions.i32
    python bench_large.py directions.i32 --only dfs --repeat 5

Each ordering runs in its own process, so its peak memory is its own and one
ordering failing to fit leaves the others standing. One JSON line per ordering.
The ``walk`` row builds the dense upstream matrix, ``n * d * 4`` bytes; on a
billion-cell raster that is tens of GiB.
"""

import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np
from numba import njit

from pyflwdir import core

MV = np.int32(-1)
METHODS = ["sort", "walk", "walk + CSR", "dfs", "topo"]


@njit
def _walk_dense(idxs_ds, idxs_pit, mv):
    """core.idxs_seq as it is on main: over the dense upstream matrix."""
    i, j = 0, 0
    idxs_us = core.upstream_matrix(idxs_ds, mv=mv)
    idxs_seq = np.full(idxs_ds.size, mv, idxs_ds.dtype)
    for idx in idxs_pit:
        idxs_seq[j] = idx
        j += 1
    while i < idxs_seq.size:
        idx0 = idxs_seq[i]
        if idx0 == mv:
            break
        for idx in idxs_us[idx0, :]:
            if idx == mv:
                break
            idxs_seq[j] = idx
            j += 1
        i += 1
    return idxs_seq[:i]


def build(label, idxs_ds, idxs_pit):
    if label == "sort":
        ranks, n = core.rank(idxs_ds, mv=MV)
        return np.argsort(ranks)[-n:].astype(idxs_ds.dtype)
    if label == "walk":
        return _walk_dense(idxs_ds, idxs_pit, MV)
    if label == "walk + CSR":
        return core.idxs_seq(idxs_ds, idxs_pit, MV)
    if label == "dfs":
        return core.idxs_seq_dfs(idxs_ds, idxs_pit, MV)
    return core.idxs_seq_topo(idxs_ds, MV)


def one(path, label, repeat):
    from pyflwdir import streams

    tiny = np.array([1, 1, -1], dtype=np.int32)  # numba compiles here,
    build(label, tiny, core.pit_indices(tiny))  # before anything is timed
    streams.accuflux(
        tiny, core.idxs_seq(tiny, core.pit_indices(tiny), MV), np.ones(3), -9999.0
    )

    idxs_ds = np.fromfile(path, dtype=np.int32)
    idxs_pit = core.pit_indices(idxs_ds)
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    t_build = min(_once(lambda: build(label, idxs_ds, idxs_pit)) for _ in range(repeat))
    seq = build(label, idxs_ds, idxs_pit)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - base

    data = np.ones(idxs_ds.size, np.float64)
    t_acc = min(
        _once(lambda: streams.accuflux(idxs_ds, seq, data, -9999.0))
        for _ in range(repeat)
    )
    print(
        json.dumps(
            dict(
                ordering=label,
                cells=int(idxs_ds.size),
                sequence=int(seq.size),
                build_s=round(t_build, 2),
                accuflux_s=round(t_acc, 2),
                peak_rss_gib=round(peak / 2**30, 2),
            )
        )
    )


def _once(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("i32", help="flat int32 next-downstream indices")
    ap.add_argument("--only", choices=METHODS)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    if a.only:
        return one(a.i32, a.only, a.repeat)
    for label in METHODS:
        r = subprocess.run(
            [
                sys.executable,
                __file__,
                a.i32,
                "--only",
                label,
                "--repeat",
                str(a.repeat),
            ]
        )
        if r.returncode != 0:
            print(json.dumps(dict(ordering=label, failed=True)))


if __name__ == "__main__":
    main()
