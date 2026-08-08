"""
Test COOToCSR conversion on Ascend NPU.

Verifies that CSRMatrix produced from COO on NPU matches the CPU reference.
Covers sorted/unsorted COO, with/without data, int32/int64, and edge cases.
"""
import sys
import pytest
import torch
import dgl
from dgl.sparse import from_coo


def _check_npu_available():
    return hasattr(torch, 'npu') and torch.npu.is_available()


TEST_CASES = [
    ("sorted_basic",       [0, 0, 1, 2], [1, 2, 0, 3], (3, 4), True),
    ("unsorted_basic",     [1, 0, 2, 0], [0, 1, 3, 2], (3, 4), False),
    ("single_row",         [0, 0, 0],     [0, 1, 2],    (1, 3), True),
    ("same_row",           [2, 2, 2],     [0, 1, 2],    (5, 3), True),
    ("empty_trailing",     [0, 1],        [0, 1],      (5, 5), True),
    ("unsorted_gaps",      [4, 0, 2, 0],  [1, 0, 3, 2], (5, 4), False),
    ("single_entry",       [0],           [0],         (1, 1), True),
    ("reverse_sorted",     [3, 2, 1, 0],  [0, 1, 2, 3], (4, 4), False),
]


def _run_coo_to_csr(rows, cols, vals, shape, device, cpu_device):
    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device), shape)
    indptr_npu, indices_npu, _ = mat_npu.csr()

    mat_cpu = from_coo(rows.to(cpu_device), cols.to(cpu_device), vals.to(cpu_device), shape)
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    return indptr_npu, indices_npu, indptr_cpu, indices_cpu


def _check(name, indptr_npu, indices_npu, indptr_cpu, indices_cpu, vals_npu=None, vals_cpu=None):
    assert torch.allclose(indptr_npu.cpu(), indptr_cpu), f"{name}: indptr mismatch"
    assert torch.allclose(indices_npu.cpu(), indices_cpu), f"{name}: indices mismatch"
    if vals_npu is not None and vals_cpu is not None:
        assert torch.allclose(vals_npu.cpu(), vals_cpu.cpu()), f"{name}: values mismatch"


# ─── Setup ─────────────────────────────────────────────────────

def _setup():
    if not _check_npu_available():
        return None, None
    device = torch.device('npu:0')
    cpu = torch.device('cpu')
    return device, cpu


# ─── Basic tests ───────────────────────────────────────────────

def test_coo_to_csr_npu_basic():
    """Sorted COO with data, int64 indices."""
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    cols = torch.tensor([1, 2, 0, 3], dtype=torch.int64)
    vals = torch.randn(4, 8)

    indptr_npu, indices_npu, indptr_cpu, indices_cpu = \
        _run_coo_to_csr(rows, cols, vals, (3, 4), device, cpu)

    _check("basic", indptr_npu, indices_npu, indptr_cpu, indices_cpu)


def test_coo_to_csr_npu_with_data():
    """COO with non-trivial data values."""
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    cols = torch.tensor([1, 2, 0, 3], dtype=torch.int64)
    vals = torch.tensor([10, 20, 30, 40], dtype=torch.float32)

    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device), (3, 4))
    mat_cpu = from_coo(rows.to(cpu), cols.to(cpu), vals.to(cpu), (3, 4))

    assert torch.allclose(mat_npu.val.cpu(), mat_cpu.val), f"with_data: NPU={mat_npu.val.cpu().tolist()}, CPU={mat_cpu.val.tolist()}"


def test_coo_to_csr_npu_without_data():
    """COO without data array (C++ auto-creates Range(0, nnz))."""
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    cols = torch.tensor([1, 2, 0, 3], dtype=torch.int64)

    mat_npu = from_coo(rows.to(device), cols.to(device), shape=(3, 4))
    indptr_npu, indices_npu, data_npu = mat_npu.csr()

    mat_cpu = from_coo(rows.to(cpu), cols.to(cpu), shape=(3, 4))
    indptr_cpu, indices_cpu, data_cpu = mat_cpu.csr()

    _check("without_data", indptr_npu, indices_npu, indptr_cpu, indices_cpu,
           data_npu, data_cpu)


# ─── Edge-ID (data field) correctness ──────────────────────────
# These tests assert the CSR matrix `data` field (edge IDs) is correct after
# COOToCSR. Previously the suite only checked indptr/indices/vals, so a bug
# where the no-data branch forgot to gather the synthesized Range(0,nnz) by the
# sort permutation (leaving data=identity while rows/cols were reordered,
# corrupting the position→eid mapping) went undetected. That mapping is what
# weighted in-edge neighbor sampling relies on (prob[data[pos]]).

# Unsorted COO with nodes that have multiple in-edges → sort perm is
# non-identity, so a wrong data array is observable. (rows, cols, shape).
EID_CASES = [
    ("unsorted_multi_in",     [0, 0, 0, 1, 1], [1, 2, 3, 0, 2], (4, 4)),
    ("unsorted_reorder",      [2, 0, 1, 0],     [0, 1, 3, 2],    (3, 4)),
    ("reverse_all",           [3, 2, 1, 0],     [0, 1, 2, 3],    (4, 4)),
    ("shuffled_large",        [4, 0, 2, 0, 3, 1, 4, 2],
                              [1, 0, 3, 2, 0, 4, 2, 1], (5, 5)),
]


def _check_csr_data(name, rows, cols, shape, dtype, device, cpu):
    """Build COO→CSR on NPU and CPU; assert indptr, indices, AND data match."""
    rows_t = torch.tensor(rows, dtype=dtype)
    cols_t = torch.tensor(cols, dtype=dtype)
    mat_npu = from_coo(rows_t.to(device), cols_t.to(device), shape=shape)
    mat_cpu = from_coo(rows_t.to(cpu), cols_t.to(cpu), shape=shape)
    indptr_n, indices_n, data_n = mat_npu.csr()
    indptr_c, indices_c, data_c = mat_cpu.csr()
    assert torch.equal(indptr_n.cpu(), indptr_c), f"{name}: indptr mismatch"
    assert torch.equal(indices_n.cpu(), indices_c), f"{name}: indices mismatch"
    assert torch.equal(data_n.cpu(), data_c), \
        f"{name}: data(eid) mismatch\n  npu={data_n.cpu().tolist()}\n  cpu={data_c.tolist()}"


@pytest.mark.parametrize("name,rows,cols,shape", [pytest.param(*c, id=c[0]) for c in EID_CASES])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_coo_to_csr_npu_data_field(name, rows, cols, shape, dtype):
    """CSR `data` (edge-ID) field must match CPU after COOToCSR reordering."""
    device, cpu = _setup()
    if device is None:
        return
    _check_csr_data(name, rows, cols, shape, dtype, device, cpu)


def test_coo_to_csr_npu_in_edges_eid():
    """Graph-level: in_edges(form='eid') must return the same edge IDs as CPU.

    This is the user-facing path that weighted in-edge sampling depends on.
    Exercises the CSC matrix (in-CSR) data array, built via COOToCSR.
    """
    device, cpu = _setup()
    if device is None:
        return

    # Unsorted COO; several nodes with multiple in-edges.
    src = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64)
    dst = torch.tensor([1, 2, 3, 0, 2, 3], dtype=torch.int64)
    g_npu = dgl.graph((src.to(device), dst.to(device)))
    g_cpu = dgl.graph((src, dst))

    for n in range(g_cpu.num_nodes()):
        eid_cpu = g_cpu.in_edges(torch.tensor([n]), form="eid").tolist()
        eid_npu = g_npu.in_edges(torch.tensor([n], device=device), form="eid").cpu().tolist()
        assert eid_npu == eid_cpu, \
            f"in_edges({n}, eid): npu={eid_npu} cpu={eid_cpu}"

    # Also verify the full edge-id set round-trips through find_edges.
    eids = torch.arange(g_cpu.num_edges())
    s_n, d_n = g_npu.find_edges(eids.to(device))
    s_c, d_c = g_cpu.find_edges(eids)
    assert torch.equal(s_n.cpu(), s_c), "find_edges src mismatch"
    assert torch.equal(d_n.cpu(), d_c), "find_edges dst mismatch"


def test_coo_to_csr_npu_data_field_random_large():
    """Random unsorted COO: CSR data must match CPU (regression for the
    no-data-gather bug on large inputs)."""
    device, cpu = _setup()
    if device is None:
        return

    torch.manual_seed(0)
    num_rows, num_cols, nnz = 500, 300, 5000
    rows = torch.randint(0, num_rows, (nnz,))
    cols = torch.randint(0, num_cols, (nnz,))
    for dtype in (torch.int32, torch.int64):
        _check_csr_data(f"random_large_{dtype}", rows.tolist(), cols.tolist(),
                        (num_rows, num_cols), dtype, device, cpu)


# ─── int32 parametrized ────────────────────────────────────────

@pytest.mark.parametrize(
    "name,rows,cols,shape,row_sorted",
    [pytest.param(n, r, c, s, f, id=n) for n, r, c, s, f in TEST_CASES],
)
def test_coo_to_csr_npu_int32(name, rows, cols, shape, row_sorted):
    device, cpu = _setup()
    if device is None:
        return

    rows_t = torch.tensor(rows, dtype=torch.int32)
    cols_t = torch.tensor(cols, dtype=torch.int32)
    vals = torch.randn(len(rows))

    indptr_npu, indices_npu, indptr_cpu, indices_cpu = \
        _run_coo_to_csr(rows_t, cols_t, vals, shape, device, cpu)

    _check(name, indptr_npu, indices_npu, indptr_cpu, indices_cpu)


# ─── int64 parametrized ────────────────────────────────────────

@pytest.mark.parametrize(
    "name,rows,cols,shape,row_sorted",
    [pytest.param(n, r, c, s, f, id=n) for n, r, c, s, f in TEST_CASES],
)
def test_coo_to_csr_npu_int64(name, rows, cols, shape, row_sorted):
    device, cpu = _setup()
    if device is None:
        return

    rows_t = torch.tensor(rows, dtype=torch.int64)
    cols_t = torch.tensor(cols, dtype=torch.int64)
    vals = torch.randn(len(rows))

    indptr_npu, indices_npu, indptr_cpu, indices_cpu = \
        _run_coo_to_csr(rows_t, cols_t, vals, shape, device, cpu)

    _check(name, indptr_npu, indices_npu, indptr_cpu, indices_cpu)


# ─── Large random test ─────────────────────────────────────────

def test_coo_to_csr_npu_random_large():
    """Random COO with 10K entries, unsorted, int32."""
    device, cpu = _setup()
    if device is None:
        return

    num_rows, num_cols = 1000, 500
    nnz = 10000
    rows_t = torch.randint(0, num_rows, (nnz,), dtype=torch.int32)
    cols_t = torch.randint(0, num_cols, (nnz,), dtype=torch.int32)
    vals = torch.randn(nnz)

    indptr_npu, indices_npu, indptr_cpu, indices_cpu = \
        _run_coo_to_csr(rows_t, cols_t, vals, (num_rows, num_cols), device, cpu)

    _check("random_large", indptr_npu, indices_npu, indptr_cpu, indices_cpu)


# ─── Graph path test ────────────────────────────────────────────

def test_coo_to_csr_npu_graph_path():
    """COOToCSR via dgl.graph path (internal COO->CSR conversion)."""
    device, cpu = _setup()
    if device is None:
        return

    src = torch.tensor([0, 1, 2, 3])
    dst = torch.tensor([1, 2, 3, 0])

    g_npu = dgl.graph((src.to(device), dst.to(device)))
    adj_npu = g_npu.adjacency_matrix()

    g_cpu = dgl.graph((src, dst))
    adj_cpu = g_cpu.adjacency_matrix()

    indptr_npu, indices_npu, _ = adj_npu.csr()
    indptr_cpu, indices_cpu, _ = adj_cpu.csr()

    _check("graph_path", indptr_npu, indices_npu, indptr_cpu, indices_cpu)
    assert g_npu.device == device


if __name__ == "__main__":
    failures = 0
    named_tests = [
        ("basic", test_coo_to_csr_npu_basic),
        ("with_data", test_coo_to_csr_npu_with_data),
        ("without_data", test_coo_to_csr_npu_without_data),
    ]
    for name, rows, cols, shape, row_sorted in TEST_CASES:
        named_tests.append(
            (f"{name}_int32", lambda n=name, r=rows, c=cols, s=shape, f=row_sorted:
             test_coo_to_csr_npu_int32(n, r, c, s, f)))
        named_tests.append(
            (f"{name}_int64", lambda n=name, r=rows, c=cols, s=shape, f=row_sorted:
             test_coo_to_csr_npu_int64(n, r, c, s, f)))
    named_tests.append(("random_large", test_coo_to_csr_npu_random_large))
    named_tests.append(("graph_path", test_coo_to_csr_npu_graph_path))
    # Edge-ID (data field) coverage — would have caught the no-data-gather bug.
    for name, rows, cols, shape in EID_CASES:
        for dtype, dname in ((torch.int32, "int32"), (torch.int64, "int64")):
            named_tests.append((
                f"data_field_{name}_{dname}",
                lambda n=name, r=rows, c=cols, s=shape, dt=dtype:
                test_coo_to_csr_npu_data_field(n, r, c, s, dt)))
    named_tests.append(("in_edges_eid", test_coo_to_csr_npu_in_edges_eid))
    named_tests.append(("data_field_random_large", test_coo_to_csr_npu_data_field_random_large))
    for name, test in named_tests:
        try:
            test()
            print(f"  PASS [{name}]")
        except AssertionError as e:
            print(f"  FAIL [{name}] {e}")
            failures += 1
    total = len(named_tests)
    print(f"\nResults: {total - failures}/{total} passed, {failures} failed", flush=True)

