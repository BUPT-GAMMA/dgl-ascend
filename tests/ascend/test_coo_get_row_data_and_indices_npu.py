"""Check the single-row COO operator and time the same call on CPU and NPU.

Run from the repository root:

    PYTHONPATH=python python tests/ascend/test_coo_get_row_data_and_indices_npu.py

Timing uses three warmups and the median of five ten-call batches, synchronized
at each batch boundary. Every timed call queries row 0, whose matching edges
occupy the first min(NNZ, 1024) COO positions.
"""

import torch

from test_coo_get_data_and_indices_npu import (
    DTYPES,
    STANDARD_NNZ,
    _benchmark,
    _npu_is_available,
    dgl,
)
from dgl import backend as F
from dgl._ffi.function import get_global_func


NUM_NODES = 4096
OUT_EDGES = get_global_func("heterograph_index._CAPI_DGLHeteroOutEdges_1")


def _invoke(graph, row=0):
    # The array-valued OutEdges entry calls COOSliceRows instead.
    return OUT_EDGES(graph._graph, graph.get_etype_id(None), row)


def _graph(rows, cols, num_nodes=NUM_NODES):
    return dgl.graph(
        (rows, cols), num_nodes=num_nodes, idtype=rows.dtype
    ).formats("coo")


def _check(packed, expected):
    for i, ref in enumerate(expected):
        actual = F.from_dgl_nd(packed(i)).cpu()
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)


def _check_edge_cases(dtype, device):
    # Multiple tiles per core, an uneven final tile, and duplicates.
    eid = torch.arange(262147, dtype=dtype)
    cols = (eid * 17 + 23) % NUM_NODES
    rows = torch.ones_like(eid)
    rows[::7] = 0
    rows[-1] = 0
    cases = [
        (rows, cols, 0, NUM_NODES),  # sparse matches across all tiles
        (rows, cols, 2, NUM_NODES),  # empty query result
        (rows[:67], cols[:67], 1, NUM_NODES),  # dense, unaligned
        (rows[:0], cols[:0], 0, NUM_NODES),  # empty graph
    ]
    # Few hits per core, including adjacent mask words and the final edge.
    sparse_rows = torch.ones_like(eid)
    sparse_rows[63::1024] = 0
    sparse_rows[64::1024] = 0
    sparse_rows[-1] = 0
    cases.append((sparse_rows, cols, 0, NUM_NODES))
    if dtype == torch.int64:
        # Equal low halves alone must not create false positives. Columns
        # also carry high bits, so the gather must preserve both halves.
        target = (1 << 32) + 13
        wide_rows = torch.tensor([13, target, target + 1, target], dtype=dtype)
        wide_cols = torch.tensor([2, target + 2, 4, target + 4], dtype=dtype)
        cases.append((wide_rows, wide_cols, target, target + 5))

        # Match both 32-bit halves of the same ID across word and tile boundaries.
        target = (1 << 32) + (1 << 31) + 13
        positions = torch.arange(100003, dtype=dtype)
        candidates = torch.tensor(
            [target, target + 1, target ^ (1 << 32)], dtype=dtype
        )
        wide_rows = candidates[positions % 3]
        wide_cols = target + positions
        cases.append((wide_rows, wide_cols, target, target + len(positions)))

    for src, dst, row, num_nodes in cases:
        positions = (src == row).nonzero().flatten()
        expected = (src[positions], dst[positions], positions.to(dtype))
        graph = _graph(src, dst, num_nodes).to(device)
        result = _invoke(graph, row)
        torch.npu.synchronize(device)
        _check(result, expected)


def test_single_row_coo_npu():
    import pytest

    if not _npu_is_available():
        pytest.skip("Ascend NPU is unavailable")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    for dtype in DTYPES.values():
        _check_edge_cases(dtype, device)


def _run_benchmark(dtype_name, dtype, device):
    for nnz in STANDARD_NNZ:
        degree = min(nnz, 1024)
        rows = torch.zeros(nnz, dtype=dtype)
        rows[degree:] = torch.arange(nnz - degree) % (NUM_NODES - 1) + 1
        cols = torch.arange(nnz, dtype=dtype) % NUM_NODES
        cpu_graph = _graph(rows, cols)
        npu_graph = cpu_graph.to(device)
        expected = (rows[:degree], cols[:degree], torch.arange(degree, dtype=dtype))
        synchronize = lambda: torch.npu.synchronize(device)

        for graph, sync in ((cpu_graph, lambda: None), (npu_graph, synchronize)):
            result = _invoke(graph)
            sync()
            _check(result, expected)

        cpu, cpu_result = _benchmark(lambda: _invoke(cpu_graph), lambda: None, 3, 10, 5)
        npu, npu_result = _benchmark(lambda: _invoke(npu_graph), synchronize, 3, 10, 5)
        _check(cpu_result, expected)
        _check(npu_result, expected)
        print(
            f"RESULT dtype={dtype_name} nnz={nnz} degree={degree} correctness=PASS "
            f"cpu_median_ms={cpu.median_ms:.6f} npu_median_ms={npu.median_ms:.6f} "
            f"npu_min_ms={npu.minimum_ms:.6f} npu_max_ms={npu.maximum_ms:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    if not _npu_is_available():
        raise RuntimeError("Ascend NPU is unavailable")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    for dtype_name, dtype in DTYPES.items():
        _run_benchmark(dtype_name, dtype, device)
        _check_edge_cases(dtype, device)
        print(f"EDGE_CASES dtype={dtype_name} correctness=PASS", flush=True)
