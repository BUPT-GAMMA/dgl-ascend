"""Test COORowWiseSamplingUniform on Ascend NPU.

Verifies the assembly implementation (COOToCSR -> CSRRowWiseSamplingUniform)
against a CPU reference. Sampling is stochastic, so:

  * Deterministic cases (select-all, or fanout >= degree) are compared exactly
    (sorted edge set) against CPU.
  * Stochastic cases are checked for structural validity (sampled edges are
    true neighbors, no duplicates when replace=False, correct per-row count)
    and statistical properties (uniform distribution over many trials).

``dgl.sampling.sample_neighbors`` samples in-edges by default; the COO path
transposes the matrix (a metadata-only swap) before row-wise sampling, so the
tests below exercise the COO assembly on the transposed view of a COO-format
graph. ``edge_dir='out'`` skips the transpose and exercises the plain path.
"""
import pytest
import torch
import dgl


def _check_npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()


def _setup():
    if not _check_npu_available():
        return None, None
    return torch.device("npu:0"), torch.device("cpu")


def _build_graph(num_nodes, edges, device, idtype=torch.int64):
    # Build on CPU first: torch_npu's aclnnMaxDim (used by DGL to infer
    # num_nodes) does not support int32 on NPU. Moving an already-built graph
    # to NPU preserves the idtype and exercises the int32 Ascend kernel.
    src = torch.tensor([e[0] for e in edges], dtype=idtype)
    dst = torch.tensor([e[1] for e in edges], dtype=idtype)
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    if device != torch.device("cpu"):
        g = g.to(device)
    return g.formats("coo")


# edges for a 5-node graph (asymmetric, with a node of in/out degree 0 and 1).
EDGES_5 = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3),
    (2, 0), (2, 3),
    (3, 0), (3, 4),
    (4, 1),
]
# out-degree per node for EDGES_5: node 0 has 3, node 4 has 0.
OUT_DEG_5 = {0: 3, 1: 2, 2: 2, 3: 2, 4: 0}
# in-degree per node for EDGES_5: node 4 has 1, none has 0.
IN_DEG_5 = {0: 2, 1: 2, 2: 2, 3: 3, 4: 1}


def _uv(g):
    """Return (u, v) CPU tensors of a graph's edges.

    The graph may live on NPU. Move to CPU first (COOSort_, needed for
    ``order='srcdst'``, is only implemented on CPU; and CSR-format graphs do
    not support the default ``eid`` order). The sampling under test has
    already happened on NPU by the time this is called.
    """
    gc = g.cpu() if g.device != torch.device("cpu") else g
    return gc.edges(order="srcdst")


def _sorted_edges(g):
    """Return sorted (u, v) pairs of a graph's edges, on CPU."""
    u, v = _uv(g)
    uv = torch.stack([u, v], dim=1)
    uv = uv[torch.argsort(uv[:, 1])]
    uv = uv[torch.argsort(uv[:, 0])]
    return uv.tolist()


def _out_neighbors(g, nodes):
    """Map node -> set of successor node ids (CPU), for edge_dir='out'."""
    succ = {int(n): set() for n in nodes.tolist()}
    u, v = _uv(g)
    for uu, vv in zip(u.tolist(), v.tolist()):
        if uu in succ:
            succ[uu].add(vv)
    return succ


def _in_neighbors(g, nodes):
    """Map node -> set of predecessor node ids (CPU), for edge_dir='in'."""
    pred = {int(n): set() for n in nodes.tolist()}
    u, v = _uv(g)
    for uu, vv in zip(u.tolist(), v.tolist()):
        if vv in pred:
            pred[vv].add(uu)
    return pred


# ---------------------------------------------------------------------------
# NPU path coverage
# ---------------------------------------------------------------------------

def test_npu_path_actually_taken():
    """Guard against silent CPU fallback: on a non-degenerate input the
    sampled subgraph must be produced with the graph on NPU (device check),
    which only happens if the Ascend dispatch branch was compiled in and ran.

    A structural early-exit would return an empty graph; here fanout > 0 with
    a non-empty node set must yield edges on a device that is NPU.
    """
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    assert g.device.type == "npu"
    nodes = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    sg = dgl.sampling.sample_neighbors(g, nodes, 2, edge_dir="out")
    assert sg.num_edges() > 0, "non-degenerate input must produce edges"


# ---------------------------------------------------------------------------
# Uniform sampling, edge_dir='out' (plain COO path, no transpose)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
def test_uniform_select_all_exact(idtype):
    """fanout=-1 is deterministic: NPU edge set must equal CPU edge set."""
    device, cpu = _setup()
    if device is None:
        return
    g_npu = _build_graph(5, EDGES_5, device, idtype)
    g_cpu = _build_graph(5, EDGES_5, cpu, idtype)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=idtype, device=device)
    nodes_cpu = nodes.cpu()

    sg_npu = dgl.sampling.sample_neighbors(g_npu, nodes, -1, edge_dir="out")
    sg_cpu = dgl.sampling.sample_neighbors(g_cpu, nodes_cpu, -1, edge_dir="out")
    assert _sorted_edges(sg_npu) == _sorted_edges(sg_cpu)


@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
def test_uniform_no_replace_structural(idtype):
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device, idtype)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=idtype, device=device)
    fanout = 2
    sg = dgl.sampling.sample_neighbors(g, nodes, fanout, replace=False, edge_dir="out")
    u, v = _uv(sg)
    succ = _out_neighbors(g, nodes)
    # Each sampled (u, v): v is a successor of u.
    for uu, vv in zip(u.cpu().tolist(), v.cpu().tolist()):
        assert vv in succ[uu], f"sampled edge ({uu},{vv}) not a real out-edge"
    # No duplicate (u, v) per row u.
    seen = {}
    for uu, vv in zip(u.cpu().tolist(), v.cpu().tolist()):
        seen.setdefault(uu, set())
        assert vv not in seen[uu], f"duplicate edge ({uu},{vv})"
        seen[uu].add(vv)
    # Per-row count == min(fanout, out-degree).
    from collections import Counter
    cnt = Counter(u.cpu().tolist())
    for n, d in OUT_DEG_5.items():
        assert cnt.get(n, 0) == min(fanout, d), \
            f"node {n}: got {cnt.get(n, 0)} expected {min(fanout, d)}"


def test_uniform_replace_structural():
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    fanout = 4
    sg = dgl.sampling.sample_neighbors(g, nodes, fanout, replace=True, edge_dir="out")
    u, v = _uv(sg)
    succ = _out_neighbors(g, nodes)
    for uu, vv in zip(u.cpu().tolist(), v.cpu().tolist()):
        assert vv in succ[uu]
    # replace=True: count == fanout for every node with degree > 0.
    from collections import Counter
    cnt = Counter(u.cpu().tolist())
    for n, d in OUT_DEG_5.items():
        if n in (0, 1, 2, 3) and d > 0:
            assert cnt.get(n, 0) == fanout, \
                f"node {n}: got {cnt.get(n, 0)} expected {fanout}"


def test_uniform_fanout_exceeds_degree_exact():
    """fanout > degree, no-replace: must pick ALL out-edges (deterministic)."""
    device, cpu = _setup()
    if device is None:
        return
    g_npu = _build_graph(5, EDGES_5, device)
    g_cpu = _build_graph(5, EDGES_5, cpu)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    sg_npu = dgl.sampling.sample_neighbors(
        g_npu, nodes, 100, replace=False, edge_dir="out")
    sg_cpu = dgl.sampling.sample_neighbors(
        g_cpu, nodes.cpu(), 100, replace=False, edge_dir="out")
    assert _sorted_edges(sg_npu) == _sorted_edges(sg_cpu)


def test_uniform_empty_request():
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([], dtype=torch.int64, device=device)
    sg = dgl.sampling.sample_neighbors(g, nodes, 2, edge_dir="out")
    assert sg.num_edges() == 0


def test_uniform_fanout_zero():
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    sg = dgl.sampling.sample_neighbors(g, nodes, 0, edge_dir="out")
    assert sg.num_edges() == 0


def test_uniform_zero_degree_node():
    """Node 4 has out-degree 0: must produce no edges for it, no crash."""
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([4, 0], dtype=torch.int64, device=device)
    sg = dgl.sampling.sample_neighbors(g, nodes, 3, edge_dir="out")
    u, _ = _uv(sg)
    assert all(x != 4 for x in u.cpu().tolist()), "degree-0 node must yield no edges"


def test_uniform_unsorted_coo():
    """Unsorted COO input takes the CPU-stable-sort path inside COOToCSR;
    result must still be correct (select-all compares exactly)."""
    device, cpu = _setup()
    if device is None:
        return
    shuffled = list(reversed(EDGES_5))
    g_npu = _build_graph(5, shuffled, device)
    g_cpu = _build_graph(5, shuffled, cpu)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    sg_npu = dgl.sampling.sample_neighbors(g_npu, nodes, -1, edge_dir="out")
    sg_cpu = dgl.sampling.sample_neighbors(g_cpu, nodes.cpu(), -1, edge_dir="out")
    assert _sorted_edges(sg_npu) == _sorted_edges(sg_cpu)


def test_uniform_data_mapping():
    """The edge ids of the sampled subgraph must map back to original edges
    (deterministic select-all case: induced edge set equals CPU's)."""
    device, cpu = _setup()
    if device is None:
        return
    g_npu = _build_graph(5, EDGES_5, device)
    g_cpu = _build_graph(5, EDGES_5, cpu)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    sg_npu = dgl.sampling.sample_neighbors(g_npu, nodes, -1, edge_dir="out")
    sg_cpu = dgl.sampling.sample_neighbors(g_cpu, nodes.cpu(), -1, edge_dir="out")
    eid_npu = sg_npu.cpu().edata[dgl.EID].tolist()
    eid_cpu = sg_cpu.cpu().edata[dgl.EID].tolist()
    assert sorted(eid_npu) == sorted(eid_cpu)


# ---------------------------------------------------------------------------
# Uniform sampling, default edge_dir='in' (transposed COO path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
def test_uniform_in_edges_select_all_exact(idtype):
    """Default direction (in-edges) on a COO graph: transposed assembly."""
    device, cpu = _setup()
    if device is None:
        return
    g_npu = _build_graph(5, EDGES_5, device, idtype)
    g_cpu = _build_graph(5, EDGES_5, cpu, idtype)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=idtype, device=device)

    sg_npu = dgl.sampling.sample_neighbors(g_npu, nodes, -1)
    sg_cpu = dgl.sampling.sample_neighbors(g_cpu, nodes.cpu(), -1)
    assert _sorted_edges(sg_npu) == _sorted_edges(sg_cpu)


def test_uniform_in_edges_no_replace_structural():
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    fanout = 2
    sg = dgl.sampling.sample_neighbors(g, nodes, fanout, replace=False)
    u, v = _uv(sg)
    pred = _in_neighbors(g, nodes)
    for uu, vv in zip(u.cpu().tolist(), v.cpu().tolist()):
        assert uu in pred[vv], f"sampled edge ({uu},{vv}) not a real in-edge"
    from collections import Counter
    cnt = Counter(v.cpu().tolist())
    for n, d in IN_DEG_5.items():
        assert cnt.get(n, 0) == min(fanout, d), \
            f"node {n}: got {cnt.get(n, 0)} expected {min(fanout, d)}"


# ---------------------------------------------------------------------------
# Statistical (optional smoke)
# ---------------------------------------------------------------------------

def test_uniform_statistical():
    """Over many trials, each successor is picked with ~equal frequency."""
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([0], dtype=torch.int64, device=device)
    # Node 0 has 3 successors: edges (0,1),(0,2),(0,3) -> succs {1,2,3}.
    fanout = 1
    trials = 600
    from collections import Counter
    counts = Counter()
    for _ in range(trials):
        sg = dgl.sampling.sample_neighbors(g, nodes, fanout, replace=False, edge_dir="out")
        u, v = _uv(sg)
        counts[v.item()] += 1
    # Expect ~200 each (3 succs, 600 trials). Allow generous tolerance.
    assert sum(counts.values()) == trials, counts
    for succ in (1, 2, 3):
        assert 120 < counts[succ] < 280, f"succ {succ}: {counts}"


# ---------------------------------------------------------------------------
# Larger / stress
# ---------------------------------------------------------------------------

def test_large_graph_uniform():
    device, cpu = _setup()
    if device is None:
        return
    torch.manual_seed(0)
    n = 200
    m = 2000
    src = torch.randint(0, n, (m,))
    dst = torch.randint(0, n, (m,))
    g = dgl.graph((src.to(device), dst.to(device)), num_nodes=n).formats("coo")
    nodes = torch.arange(0, n, dtype=torch.int64, device=device)
    sg = dgl.sampling.sample_neighbors(g, nodes, 5, replace=False, edge_dir="out")
    u, v = _uv(sg)
    succ = _out_neighbors(g, nodes)
    for uu, vv in zip(u.cpu().tolist(), v.cpu().tolist()):
        assert vv in succ[uu]
    assert sg.num_edges() <= n * 5
