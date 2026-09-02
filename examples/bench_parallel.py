# -*- coding: utf-8 -*-
"""Parallel accuflux over a depth-first ordering: by basin, and by layer.

Two ways to split the work, from Deltares/pyflwdir#85:

* by basin: a depth-first sequence lays every basin out as one contiguous run,
  so the runs go to threads with nothing shared;
* by layer: cells are laid into layers such that all upstream cells of a cell
  sit in earlier layers AND no two cells of a layer share a downstream cell,
  so one layer runs with no locks and no atomic adds.

    python bench_parallel.py rhine_d8.tif
    python bench_parallel.py flwdir.tif --col 30000 --row 3600 --size 4000

Prints a serial baseline and both parallel timings per thread count, and checks
that every result matches the serial sweep to 1e-9 relative.
"""

import argparse
import time

import numba
import numpy as np
import rasterio
from numba import njit, prange
from rasterio.windows import Window

import pyflwdir
from pyflwdir import core, streams


@njit
def layering_conflict_free(idxs_ds, mv):
    """Layer per cell, -1 for nodata. Every upstream cell of a cell is in an
    earlier layer, and no two cells in a layer share a downstream cell."""
    n = idxs_ds.size
    layer = np.full(n, -1, np.int32)
    n_up = np.zeros(n, np.int32)
    n_valid = 0
    for i in range(n):
        ds = idxs_ds[i]
        if ds != mv:
            n_valid += 1
            if ds != i:
                n_up[ds] += 1
    done = np.zeros(n, np.uint8)
    claimed = np.zeros(n, np.uint8)  # downstream cells taken this layer
    curr = np.empty(n, np.int32)
    nxt = np.empty(n, np.int32)
    taken = np.empty(n, np.int32)
    n_curr = 0
    for i in range(n):  # start from the headwaters
        if idxs_ds[i] != mv and n_up[i] == 0:
            curr[n_curr] = i
            n_curr += 1
    lay, n_done = 0, 0
    while n_done < n_valid and n_curr > 0:
        n_nxt, n_taken = 0, 0
        for k in range(n_curr):
            idx = curr[k]
            if done[idx] == 1:
                continue
            ds = idxs_ds[idx]
            if ds == mv or ds == idx:  # a pit writes nowhere
                layer[idx] = lay
                done[idx] = 1
                n_done += 1
                continue
            if claimed[ds] == 1:  # someone in this layer writes there
                nxt[n_nxt] = idx  # so this cell waits one layer
                n_nxt += 1
                continue
            layer[idx] = lay
            done[idx] = 1
            n_done += 1
            claimed[ds] = 1
            taken[n_taken] = ds
            n_taken += 1
            n_up[ds] -= 1
            if n_up[ds] == 0 and done[ds] == 0:
                nxt[n_nxt] = ds
                n_nxt += 1
        for i in range(n_taken):  # release the claims for the next layer
            claimed[taken[i]] = 0
        curr, nxt = nxt, curr
        n_curr, lay = n_nxt, lay + 1
    return layer


@njit(parallel=True)
def accuflux_by_basin(idxs_ds, start, cells, data, nb):
    """One thread per contiguous basin run of the depth-first sequence."""
    accu = data.copy()
    for b in prange(nb):
        for i in range(start[b + 1] - 1, start[b] - 1, -1):  # up- to downstream
            idx0 = cells[i]
            idx_ds = idxs_ds[idx0]
            if idx0 != idx_ds:
                accu[idx_ds] += accu[idx0]
    return accu


@njit(parallel=True)
def accuflux_by_layer(idxs_ds, start, cells, data, nl):
    """Layers in order, the cells of one layer in parallel, no atomics."""
    accu = data.copy()
    for L in range(nl):
        for i in prange(start[L], start[L + 1]):
            idx0 = cells[i]
            idx_ds = idxs_ds[idx0]
            if idx0 != idx_ds:
                accu[idx_ds] += accu[idx0]
    return accu


def best(fn, repeat):
    fn()
    b = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif", help="a d8 flow direction raster")
    ap.add_argument("--col", type=int, default=0)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--size", type=int, default=0, help="window size, 0 = whole raster")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    a = ap.parse_args()

    win = Window(a.col, a.row, a.size, a.size) if a.size else None
    with rasterio.open(a.tif) as src:
        d = src.read(1, window=win)
        tr = src.window_transform(win) if win else src.transform
    flw = pyflwdir.from_array(d, ftype="d8", transform=tr, latlon=True)
    ids, ip, mv = flw.idxs_ds, flw.idxs_pit, flw._mv
    area = flw.area.ravel() / 1e6

    # by basin: the depth-first sequence, cut at the pits
    s = core.idxs_seq_dfs(ids, ip, mv).astype(np.int64)
    start_b = np.append(np.flatnonzero(ids[s] == s), s.size).astype(np.int64)
    nb = start_b.size - 1

    # by layer: the conflict-free layering, cells sorted by layer
    layer = layering_conflict_free(ids, mv)
    ok = layer >= 0
    cells_l = np.flatnonzero(ok).astype(np.int64)
    order = np.argsort(layer[ok], kind="stable")
    cells_l = cells_l[order]
    nl = int(layer.max()) + 1
    start_l = np.zeros(nl + 1, np.int64)
    np.cumsum(np.bincount(layer[ok], minlength=nl), out=start_l[1:])
    key = layer[cells_l] * np.int64(ids.size + 1) + ids[cells_l].astype(np.int64)
    assert (
        np.unique(key).size == key.size
    ), "two cells of one layer share a downstream cell"

    seq = core.idxs_seq(ids, ip, mv)
    ref = streams.accuflux(ids, seq, area, -9999.0)
    t0 = best(lambda: streams.accuflux(ids, seq, area, -9999.0), a.repeat)
    print(f"{ids.size:,} cells, {nb:,} basins, {nl:,} layers")
    print(f"serial accuflux over idxs_seq: {t0:.1f} ms\n")
    print(f"{'threads':>8} {'by basin':>10} {'by layer':>10}")
    for nt in a.threads:
        numba.set_num_threads(nt)
        ob = accuflux_by_basin(ids, start_b, s, area, nb)
        ol = accuflux_by_layer(ids, start_l, cells_l, area, nl)
        assert np.allclose(ob, ref, rtol=1e-9, atol=0)
        assert np.allclose(ol, ref, rtol=1e-9, atol=0)
        tb = best(lambda: accuflux_by_basin(ids, start_b, s, area, nb), a.repeat)
        tl = best(lambda: accuflux_by_layer(ids, start_l, cells_l, area, nl), a.repeat)
        print(f"{nt:>8} {tb:>8.1f} ms {tl:>8.1f} ms")


if __name__ == "__main__":
    main()
