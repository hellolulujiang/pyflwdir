# -*- coding: utf-8 -*-
"""Tests for the pyflwdir.core.py submodule."""

import pytest
import numpy as np

from pyflwdir import core, streams


@pytest.mark.parametrize(
    "test_data, flwdir", [("test_data0", "flwdir0"), ("test_data0", "flwdir0")]
)
def test_downstream(test_data, flwdir, request):
    test_data = request.getfixturevalue(test_data)
    flwdir = request.getfixturevalue(flwdir)
    idxs_ds, idxs_pit, seq, rank, mv = [p.copy() for p in test_data]
    n, ncol = np.sum(idxs_ds != mv), flwdir.shape[1]
    # rank
    assert np.sum(rank == 0) == idxs_pit.size
    if np.any(rank > 0):
        idxs_mask = np.where(rank > 0)[0]  # valid and no pits
        assert np.all(rank[idxs_mask] == rank[idxs_ds[idxs_mask]] + 1)
    # pit indices
    idxs_pit1 = np.sort(core.pit_indices(idxs_ds))
    assert np.all(idxs_pit1 == np.sort(idxs_pit))
    # loop indices
    idxs_loop = core.loop_indices(idxs_ds, mv=mv)
    assert seq.size == n - idxs_loop.size
    # local upstream indices
    if np.any(rank >= 2):
        rmax = np.max(rank)
        idxs = np.where(rank == rmax)[0]
        # path
        paths, dists = core.path(idxs, idxs_ds, mv=mv)
        assert np.all([p.size for p in paths] == rmax + 1)
        assert np.all(dists == rmax)
        # snap
        idxs1, dists1 = core.snap(idxs, idxs_ds, ncol, real_length=True, mv=mv)
        assert np.all([idxs_ds[idx] == idx for idx in idxs1])
        assert np.all(dists1 >= rmax)
        idxs2, dists2 = core.snap(idxs, idxs_ds, real_length=False, max_length=2, mv=mv)
        assert np.all(dists2 == 2)
        assert np.all(rank[idxs2] == rmax - 2)
        idxs2, dists2 = core.snap(idxs, idxs_ds, mask=rank <= rmax - 2, mv=mv)
        assert np.all(dists2 == 2)
        assert np.all(rank[idxs2] == rmax - 2)
        # window
        idx0 = np.where(rank == 2)[0][0]
        path = core._trace(idx0, idxs_ds, mv=mv)[0]
        wdw = core._window(idx0, 2, idxs_ds, idxs_ds, mv=mv)
        assert np.all(path == wdw[2:]) and np.all(path[::-1] == wdw[:-2])
        ##
        rank1 = core.fillnodata_downstream(idxs_ds, seq, rank, nodata=0)
        idxs1 = idxs_ds[np.where(rank == 1)[0]]
        idxs1, n_up = np.unique(idxs1, return_counts=True)
        assert np.all(rank1[idxs1] == 1)
        rank2 = core.fillnodata_downstream(idxs_ds, seq, rank, nodata=0, how="min")
        assert np.all(rank2 == rank1)
        rank3 = core.fillnodata_downstream(idxs_ds, seq, rank, nodata=0, how="sum")
        assert np.all(rank3[idxs1] == n_up)


def test_idxs_seq_orderings_nodata_target():
    # cell 1 drains into cell 2, which is itself nodata: walk and dfs leave
    # cell 1 out, topo keeps it, and nobody may touch cell 2 or corrupt counts
    idxs_ds = np.array([0, 2, -1, 0], dtype=np.int32)  # 0 pit, 3 -> 0
    mv = np.int32(-1)
    idxs_pit = core.pit_indices(idxs_ds)
    walk = core.idxs_seq(idxs_ds, idxs_pit, mv)
    dfs = core.idxs_seq_dfs(idxs_ds, idxs_pit, mv)
    topo = core.idxs_seq_topo(idxs_ds, mv)
    assert np.array_equal(np.sort(walk), [0, 3])
    assert np.array_equal(np.sort(dfs), [0, 3])
    assert np.array_equal(np.sort(topo), [0, 1, 3])
    assert 2 not in topo


@pytest.mark.parametrize("test_data", ["test_data0", "test_data1", "test_data2"])
def test_idxs_seq_orderings(test_data, request):
    test_data = request.getfixturevalue(test_data)
    idxs_ds, idxs_pit, seq, rank, mv = [p.copy() for p in test_data]
    idxs_ds[rank == -1] = mv
    n = idxs_ds.size
    seqs = {
        "walk": core.idxs_seq(idxs_ds, idxs_pit, mv=mv),
        "dfs": core.idxs_seq_dfs(idxs_ds, idxs_pit, mv=mv),
        "topo": core.idxs_seq_topo(idxs_ds, mv=mv),
    }
    nonpit = (idxs_ds != mv) & (idxs_ds != np.arange(n, dtype=idxs_ds.dtype))
    idxs_up = np.where(nonpit)[0]
    for name, seq1 in seqs.items():
        # the same cells as the sorted sequence of the fixture
        assert np.array_equal(np.sort(seq1), np.sort(seq)), name
        # every cell comes after the cell it drains into
        pos = np.full(n, -1, np.int64)
        pos[seq1] = np.arange(seq1.size)
        assert np.all(pos[idxs_ds[idxs_up]] < pos[idxs_up]), name


@pytest.mark.parametrize("test_data", ["test_data0", "test_data1", "test_data2"])
def test_upstream_csr(test_data, request):
    test_data = request.getfixturevalue(test_data)
    idxs_ds, idxs_pit, seq, rank, mv = [p.copy() for p in test_data]
    idxs_ds[rank == -1] = mv
    n = idxs_ds.size
    indptr, idxs_us = core.upstream_csr(idxs_ds, mv=mv)
    # one slice per cell, one entry per upstream cell
    assert indptr.size == n + 1
    assert indptr[0] == 0
    assert indptr[n] == idxs_us.size == seq.size - idxs_pit.size
    assert np.all(np.diff(indptr.astype(np.int64)) >= 0)
    # every entry drains into the cell whose slice it sits in
    for idx0 in range(n):
        for k in range(indptr[idx0], indptr[idx0 + 1]):
            assert idxs_ds[idxs_us[k]] == idx0
    # same upstream cells, in the same order, as the dense upstream matrix
    idxs_us0 = core.upstream_matrix(idxs_ds, mv=mv)
    for idx0 in range(n):
        us0 = idxs_us0[idx0][idxs_us0[idx0] != mv]
        assert np.array_equal(idxs_us[indptr[idx0] : indptr[idx0 + 1]], us0)


@pytest.mark.parametrize(
    "test_data, flwdir", [("test_data0", "flwdir0"), ("test_data0", "flwdir0")]
)
def test_upstream(test_data, flwdir, request):
    test_data = request.getfixturevalue(test_data)
    flwdir = request.getfixturevalue(flwdir)
    idxs_ds, idxs_pit, seq, rank, mv = [p.copy() for p in test_data]
    idxs_ds[rank == -1] = mv
    n, ncol = np.sum(idxs_ds != mv), flwdir.shape[1]
    upa = streams.upstream_area(idxs_ds, seq, ncol, dtype=np.int32)
    # count
    n_up = core.upstream_count(idxs_ds, mv=mv)
    assert np.sum(n_up[n_up != -9]) == n - idxs_pit.size
    # upstream matrix
    idxs_us = core.upstream_matrix(idxs_ds, mv=mv)
    assert np.sum(idxs_us != mv) == seq.size - idxs_pit.size
    # ordered
    seq2 = core.idxs_seq(idxs_ds, idxs_pit, mv=mv)
    assert np.all(np.diff(rank.flat[seq2]) >= 0)
    # headwater
    idxs_headwater = core.headwater_indices(idxs_ds, mv=mv)
    assert np.all(n_up[idxs_headwater] == 0)
    if np.any(n_up > 0):
        # local upstream indices
        idx0 = np.where(upa == np.max(upa))[0][0]
        idxs_us0 = np.sort(core._upstream_d8_idx(idx0, idxs_ds, flwdir.shape))
        idxs_us1 = np.sort(idxs_us[idx0, : n_up[idx0]])
        assert np.all(idxs_us1 == idxs_us0)
        # main upstream
        idxs_us_main = core.main_upstream(idxs_ds, upa, mv=mv)
        assert np.any(idxs_us0 == idxs_us_main[idx0])
        idxs = np.where(idxs_us_main != mv)[0]
        assert np.all(idxs_ds[idxs_us_main[idxs]] == idxs)
        assert idxs.size == np.sum(n_up[upa > 0] >= 1)
        # window
        path = core._trace(idx0, idxs_us_main, ncol, mv=mv)[0]
        wdw = core._window(idx0, 1, idxs_us_main, idxs_us_main, mv=mv)
        assert np.all(path[:2] == wdw[1:]) and np.all(path[:2][::-1] == wdw[:-1])
        # # tributary
        # idxs_us_trib = core.main_tributary(idxs_ds, idxs_us_main, upa, mv=mv)
        # idxs = np.where(idxs_us_trib != mv)[0]
        # assert idxs.size == np.sum(n_up[upa > 0] > 1)
        # if idxs.size > 0:
        #     assert np.all(idxs_ds[idxs_us_main[idxs]] == idxs)
        # # tributaries
        # idxs_trib = core._tributaries(idx0, idxs_us_main, idxs_us_trib, upa, mv=mv)
        # assert np.all([np.any(idx == idxs_us_trib[path]) for idx in idxs_trib])
        # if idxs_trib.size > 1:
        #     idxs_trib1 = core._tributaries(
        #         idx0, idxs_us_main, idxs_us_trib, upa, n=1, mv=mv
        #     )
        #     assert np.max(upa[idxs_trib]) == upa[idxs_trib1]
