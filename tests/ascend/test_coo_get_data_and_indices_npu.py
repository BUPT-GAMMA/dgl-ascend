"""COOGetDataAndIndices graph-API regression on Ascend.

Run with DGLBACKEND=pytorch and an Ascend-enabled DGL build. COO is forced so
edge_ids(return_uv=True) exercises UnitGraph COO EdgeIdsAll. Explicit COO data,
large coordinates and invalid low-level arrays are covered by the C++ tests.
Every comparison preserves the original three-array order; no sort or xfail.
"""

import pytest
import torch

pytest.importorskip("torch_npu", reason="Ascend tests require torch_npu")

import dgl


@pytest.fixture(scope="module")
def npu_device():
    if not torch.npu.is_available():
        pytest.skip("Ascend device unavailable; this is not a passing NPU run")
    return torch.device("npu:0")


def _make_graph(dtype, nnz):
    positions = torch.arange(nnz, dtype=torch.int64)
    src = (positions % 7).to(dtype)
    dst = ((positions // 7) % 5).to(dtype)
    return dgl.graph(
        (src, dst), num_nodes=7, idtype=dtype
    ).formats("coo")


def _queries(dtype, count, mode):
    positions = torch.arange(count, dtype=torch.int64)
    rows = (positions % 7).to(dtype)
    cols = ((positions // 7) % 5).to(dtype)
    if mode == "rows_broadcast":
        rows = torch.tensor([2], dtype=dtype)
    elif mode == "cols_broadcast":
        cols = torch.tensor([3], dtype=dtype)
    return rows, cols


def _check_outputs(cpu_result, npu_result, dtype, device):
    assert len(cpu_result) == len(npu_result) == 3
    for name, expected, actual in zip(
        ("ret_rows", "ret_cols", "ret_data"), cpu_result, npu_result
    ):
        assert actual.device == device, name
        assert actual.dtype == expected.dtype == dtype, name
        assert actual.ndim == expected.ndim == 1, name
        assert actual.shape == expected.shape, name
        assert torch.equal(actual.cpu(), expected), name


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("nnz", [0, 1, 513])
@pytest.mark.parametrize("count", [0, 1, 199, 200, 201])
@pytest.mark.parametrize(
    "mode", ["paired", "rows_broadcast", "cols_broadcast"]
)
def test_coo_edge_ids_match_native_cpu(npu_device, dtype, nnz, count, mode):
    cpu_graph = _make_graph(dtype, nnz)
    npu_graph = cpu_graph.to(npu_device)
    assert cpu_graph.formats()["created"] == ["coo"]
    assert npu_graph.formats()["created"] == ["coo"]
    rows, cols = _queries(dtype, count, mode)

    expected = cpu_graph.edge_ids(rows, cols, return_uv=True)
    actual = npu_graph.edge_ids(
        rows.to(npu_device), cols.to(npu_device), return_uv=True
    )
    _check_outputs(expected, actual, dtype, npu_device)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_coo_edge_ids_reject_out_of_bounds(npu_device, dtype):
    cpu_graph = _make_graph(dtype, 513)
    npu_graph = cpu_graph.to(npu_device)
    rows = torch.tensor([7], dtype=dtype)
    cols = torch.tensor([0], dtype=dtype)
    with pytest.raises(dgl.DGLError):
        cpu_graph.edge_ids(rows, cols, return_uv=True)
    with pytest.raises(dgl.DGLError):
        npu_graph.edge_ids(
            rows.to(npu_device), cols.to(npu_device), return_uv=True
        )
