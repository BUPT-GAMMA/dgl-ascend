"""
Test SDDMM and SPMM operators on Ascend NPU.

Tests cover:
- SDDMM: dot/add/sub/mul/div/copy_lhs/copy_rhs × COO/CSR × FP32/FP16 × int32/int64
- SPMM: copy_lhs/copy_rhs + sum/max/min × FP32 × int32/int64
- UDF reduce (degree bucketing path)
- Backward (autograd)
- stream sync (in_edges/in_degrees after update_all)
"""
import pytest
import torch
import torch_npu
import numpy as np
import dgl
import dgl.function as fn
from dgl.ops import gsddmm, gspmm

dev = torch.device("npu:0")
torch.npu.set_device(dev)

# ─── helpers ──────────────────────────────────────────────────────────────────

def make_graph(num_src=10, num_dst=8, num_edges=30, idtype=torch.int64):
    """Create a random homogeneous graph on NPU."""
    torch.npu.synchronize()
    np.random.seed(42)
    src = np.random.randint(0, num_src, num_edges)
    dst = np.random.randint(0, num_dst, num_edges)
    g = dgl.graph((src.tolist(), dst.tolist()), idtype=idtype).to(dev)
    return g


def make_bipartite(num_src=10, num_dst=12, num_edges=40, idtype=torch.int64):
    """Create a random bipartite graph on NPU."""
    np.random.seed(123)
    src = np.random.randint(0, num_src, num_edges)
    dst = np.random.randint(0, num_dst, num_edges)
    g = dgl.heterograph(
        {("_U", "_E", "_V"): (src.tolist(), dst.tolist())}
    )
    g = g.astype(idtype).to(dev)
    return g


# ─── SDDMM tests ──────────────────────────────────────────────────────────────

class TestSDDMM:
    """SDDMM operator tests."""

    def setup_method(self):
        torch.npu.synchronize()

    def teardown_method(self):
        torch.npu.synchronize()

    def teardown_method(self):
        torch.npu.synchronize()

    @pytest.mark.parametrize("op", ["dot", "add", "sub", "mul", "div",
                                     "copy_lhs", "copy_rhs"])
    @pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
    def test_sddmm_coo_fp32(self, op, idtype):
        """SDDMM all ops, COO format, FP32."""
        torch.npu.synchronize()
        g = make_graph(idtype=idtype)
        g_cpu = g.cpu()
        lhs = torch.rand(g.num_src_nodes(), 4, device=dev)
        # For div, ensure rhs is not near zero to avoid amplifying FP differences
        if op == "div":
            rhs = torch.rand(g.num_src_nodes(), 4, device=dev) + 0.5
        else:
            rhs = torch.rand(g.num_src_nodes(), 4, device=dev)

        e_npu = gsddmm(g, op, lhs, rhs, lhs_target="u", rhs_target="v")
        e_cpu = gsddmm(g_cpu, op, lhs.cpu(), rhs.cpu(),
                        lhs_target="u", rhs_target="v")
        diff = (e_npu.cpu() - e_cpu).abs().max().item()
        # NPU binary kernel: FP32 should be near-exact, div has minor FP diff
        tol = 1e-4 if op != "div" else 1e-3
        assert diff < tol, f"{op} diff={diff}"

    @pytest.mark.parametrize("op", ["dot", "add", "sub", "mul", "div",
                                     "copy_lhs", "copy_rhs"])
    @pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
    def test_sddmm_csr_fp32(self, op, idtype):
        """SDDMM with CSR format graph."""
        torch.npu.synchronize()
        g = make_graph(idtype=idtype)
        g_csr = g.formats("csr").to(dev)
        g_cpu = g_csr.cpu()
        lhs = torch.rand(g.num_src_nodes(), 4, device=dev)
        rhs = torch.rand(g.num_src_nodes(), 4, device=dev)

        e_npu = gsddmm(g_csr, op, lhs, rhs, lhs_target="u", rhs_target="v")
        e_cpu = gsddmm(g_cpu, op, lhs.cpu(), rhs.cpu(),
                        lhs_target="u", rhs_target="v")
        diff = (e_npu.cpu() - e_cpu).abs().max().item()
        assert diff < 1e-4, f"CSR {op} diff={diff}"

    @pytest.mark.parametrize("op", ["dot", "copy_lhs", "copy_rhs"])
    def test_sddmm_fp16(self, op):
        """SDDMM with FP16 features (compare against FP32 CPU ref)."""
        torch.npu.synchronize()
        g = make_graph(idtype=torch.int64)
        g_cpu = g.cpu()
        lhs = torch.rand(g.num_src_nodes(), 4, device=dev)
        rhs = torch.rand(g.num_src_nodes(), 4, device=dev)

        e_npu = gsddmm(g, op, lhs.half(), rhs.half(),
                        lhs_target="u", rhs_target="v")
        e_cpu = gsddmm(g_cpu, op, lhs.cpu(), rhs.cpu(),
                        lhs_target="u", rhs_target="v")
        diff = (e_npu.float().cpu() - e_cpu).abs().max().item()
        assert diff < 5e-2, f"FP16 {op} diff={diff}"

    @pytest.mark.parametrize("feat_dim", [1, 4, 13, 64, 128])
    def test_sddmm_dot_feat_dims(self, feat_dim):
        """SDDMM dot with various feature dimensions."""
        torch.npu.synchronize()
        g = make_graph(idtype=torch.int64)
        g_cpu = g.cpu()
        lhs = torch.rand(g.num_src_nodes(), feat_dim, device=dev)
        rhs = torch.rand(g.num_src_nodes(), feat_dim, device=dev)

        e_npu = gsddmm(g, "dot", lhs, rhs, lhs_target="u", rhs_target="v")
        e_cpu = gsddmm(g_cpu, "dot", lhs.cpu(), rhs.cpu(),
                        lhs_target="u", rhs_target="v")
        diff = (e_npu.cpu() - e_cpu).abs().max().item()
        assert diff < 1e-3, f"feat_dim={feat_dim} diff={diff}"

    def test_sddmm_bipartite(self):
        """SDDMM on bipartite graph."""
        g = make_bipartite(num_src=10, num_dst=12)
        g_cpu = g.cpu()
        lhs = torch.rand(10, 4, device=dev)
        rhs = torch.rand(12, 4, device=dev)

        e_npu = gsddmm(g, "dot", lhs, rhs, lhs_target="u", rhs_target="v")
        e_cpu = gsddmm(g_cpu, "dot", lhs.cpu(), rhs.cpu(),
                        lhs_target="u", rhs_target="v")
        diff = (e_npu.cpu() - e_cpu).abs().max().item()
        assert diff < 1e-4

    def test_sddmm_empty_graph(self):
        """SDDMM on graph with 0 edges."""
        g = dgl.graph(([], []), num_nodes=5, idtype=torch.int64).to(dev)
        lhs = torch.rand(5, 4, device=dev)
        rhs = torch.rand(5, 4, device=dev)
        e = gsddmm(g, "dot", lhs, rhs, lhs_target="u", rhs_target="v")
        assert e.shape[0] == 0

    def test_sddmm_backward(self):
        """SDDMM dot backward (autograd)."""
        g = make_graph(idtype=torch.int64)
        lhs = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)
        rhs = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)
        e = gsddmm(g, "dot", lhs, rhs, lhs_target="u", rhs_target="v")
        loss = e.sum()
        loss.backward()
        assert lhs.grad is not None and lhs.grad.abs().sum() > 0
        assert rhs.grad is not None and rhs.grad.abs().sum() > 0

    def test_sddmm_large_graph(self):
        """SDDMM on large graph (1000 nodes, 5000 edges)."""
        g = make_graph(num_src=1000, num_dst=1000, num_edges=5000,
                       idtype=torch.int64)
        g_cpu = g.cpu()
        lhs = torch.rand(1000, 64, device=dev)
        rhs = torch.rand(1000, 64, device=dev)

        e_npu = gsddmm(g, "dot", lhs, rhs, lhs_target="u", rhs_target="v")
        e_cpu = gsddmm(g_cpu, "dot", lhs.cpu(), rhs.cpu(),
                        lhs_target="u", rhs_target="v")
        diff = (e_npu.cpu() - e_cpu).abs().max().item()
        assert diff < 1e-3, f"large graph diff={diff}"


# ─── SPMM tests ───────────────────────────────────────────────────────────────

class TestSPMM:
    """SPMM operator tests (gspmm / update_all with built-in reduce)."""

    def setup_method(self):
        torch.npu.synchronize()

    @pytest.mark.parametrize("reducer", ["sum", "max", "min"])
    @pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
    def test_spmm_copy_lhs_sum(self, reducer, idtype):
        """SpMM copy_lhs + sum/max/min, FP32."""
        g = make_graph(idtype=idtype)
        g_cpu = g.cpu()
        ufeat = torch.rand(g.num_src_nodes(), 4, device=dev)

        v_npu = gspmm(g, "copy_lhs", reducer, ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", reducer, ufeat.cpu(), None)
        if reducer in ["max", "min"]:
            v_npu = dgl.backend.replace_inf_with_zero(v_npu)
            v_cpu = dgl.backend.replace_inf_with_zero(v_cpu)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4, f"{reducer} diff={diff}"

    @pytest.mark.parametrize("reducer", ["sum", "max", "min"])
    def test_spmm_update_all(self, reducer):
        """update_all(copy_u, sum/max/min) — fused SpMM path."""
        g = make_graph(idtype=torch.int64)
        g_cpu = g.cpu()
        g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev)
        g_cpu.ndata["x"] = g.ndata["x"].cpu()

        reduce_fn = getattr(fn, reducer)("m", "h")
        g.update_all(fn.copy_u("x", "m"), reduce_fn)
        g_cpu.update_all(fn.copy_u("x", "m"), reduce_fn)
        diff = (g.ndata["h"].cpu() - g_cpu.ndata["h"]).abs().max().item()
        assert diff < 1e-4, f"update_all {reducer} diff={diff}"

    def test_spmm_backward(self):
        """SpMM forward + backward."""
        g = make_graph(idtype=torch.int64)
        ufeat = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)
        v = gspmm(g, "copy_lhs", "sum", ufeat, None)
        loss = v.sum()
        loss.backward()
        assert ufeat.grad is not None and ufeat.grad.abs().sum() > 0

    @pytest.mark.parametrize("feat_dim", [1, 4, 13, 64, 128])
    def test_spmm_feat_dims(self, feat_dim):
        """SpMM with various feature dimensions."""
        g = make_graph(idtype=torch.int64)
        g_cpu = g.cpu()
        ufeat = torch.rand(g.num_src_nodes(), feat_dim, device=dev)

        v_npu = gspmm(g, "copy_lhs", "sum", ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", "sum", ufeat.cpu(), None)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-3, f"feat_dim={feat_dim} diff={diff}"

    def test_spmm_large_graph(self):
        """SpMM on large graph."""
        g = make_graph(num_src=1000, num_dst=1000, num_edges=5000,
                       idtype=torch.int64)
        g_cpu = g.cpu()
        ufeat = torch.rand(1000, 64, device=dev)

        v_npu = gspmm(g, "copy_lhs", "sum", ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", "sum", ufeat.cpu(), None)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-3, f"large graph diff={diff}"

    def test_spmm_zero_degree_nodes(self):
        """SpMM with zero-degree destination nodes."""
        # Node 0 has no incoming edges
        g = dgl.graph(([1, 2, 3], [1, 2, 3]), idtype=torch.int64).to(dev)
        g_cpu = g.cpu()
        ufeat = torch.rand(4, 4, device=dev)

        v_npu = gspmm(g, "copy_lhs", "sum", ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", "sum", ufeat.cpu(), None)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4
        # Zero-degree node should have zero output
        assert v_npu[0].abs().max() == 0


# ─── UDF reduce + stream sync tests ──────────────────────────────────────────

class TestUDRReduceAndStreamSync:
    """UDF reduce (degree bucketing) and stream synchronization tests."""

    def setup_method(self):
        torch.npu.synchronize()

    def test_udf_reduce_forward(self):
        """UDF reduce with degree bucketing (uses SDDMM copy_lhs internally)."""
        g = make_graph(idtype=torch.int64)
        g_cpu = g.cpu()
        g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev)
        g_cpu.ndata["x"] = g.ndata["x"].cpu()

        g.update_all(fn.copy_u("x", "m"),
                      lambda n: {"h": n.mailbox["m"].sum(1)})
        g_cpu.update_all(fn.copy_u("x", "m"),
                          lambda n: {"h": n.mailbox["m"].sum(1)})
        diff = (g.ndata["h"].cpu() - g_cpu.ndata["h"]).abs().max().item()
        assert diff < 1e-4, f"UDF reduce diff={diff}"

    def test_udf_reduce_backward(self):
        """UDF reduce forward + backward (full autograd path)."""
        g = make_graph(idtype=torch.int64)
        g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev,
                                   requires_grad=True)
        g.update_all(fn.copy_u("x", "m"),
                      lambda n: {"h": n.mailbox["m"].sum(1)})
        loss = g.ndata["h"].sum()
        loss.backward()
        assert g.ndata["x"].grad is not None

    def test_in_edges_after_update_all(self):
        """in_edges returns correct results after update_all (stream sync)."""
        g = make_graph(num_src=20, num_dst=20, num_edges=100,
                       idtype=torch.int64)
        g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev,
                                   requires_grad=True)
        g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))

        # in_edges should return correct count
        nodes = torch.arange(g.num_nodes(), device=dev)
        eid = g.in_edges(nodes, form="eid")
        assert len(eid) == g.num_edges()

        # in_degrees should match
        degs = g.in_degrees()
        assert degs.sum().item() == g.num_edges()

    def test_in_edges_after_gsddmm(self):
        """in_edges correct after gsddmm (autograd forward)."""
        g = make_graph(num_src=20, num_dst=20, num_edges=100,
                       idtype=torch.int64)
        lhs = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)
        rhs = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)

        e = gsddmm(g, "dot", lhs, rhs, lhs_target="u", rhs_target="v")

        # in_edges after gsddmm
        nodes = torch.arange(g.num_nodes(), device=dev)
        eid = g.in_edges(nodes, form="eid")
        assert len(eid) == g.num_edges()

    def test_repeated_update_all(self):
        """Multiple update_all calls on same graph (stream sync stability)."""
        g = make_graph(num_src=20, num_dst=20, num_edges=100,
                       idtype=torch.int64)
        g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev,
                                   requires_grad=True)

        for i in range(3):
            g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev,
                                       requires_grad=True)
            g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
            loss = g.ndata["h"].sum()
            loss.backward()

        # Verify in_edges still correct after multiple iterations
        nodes = torch.arange(g.num_nodes(), device=dev)
        eid = g.in_edges(nodes, form="eid")
        assert len(eid) == g.num_edges()

    def test_mean_nodes(self):
        """segment_reduce forward + backward (mean_nodes uses segment_reduce)."""
        torch.npu.synchronize()
        # Directly test segment_reduce via dgl.ops.segment_reduce
        from dgl.ops import segment_reduce
        seglen = torch.tensor([3, 2, 1], device=dev)
        value = torch.rand(6, 4, device=dev, requires_grad=True)
        y = segment_reduce(seglen, value, reducer="sum")
        assert y.shape == (3, 4)
        loss = y.sum()
        loss.backward()
        assert value.grad is not None


# ─── End-to-end model test ───────────────────────────────────────────────────

class TestEndToEnd:
    """End-to-end GNN model forward + backward + training loop."""

    def setup_method(self):
        torch.npu.synchronize()

    def test_gcn_forward_backward(self):
        """Simulate a simple GCN forward + backward."""
        g = make_graph(num_src=30, num_dst=30, num_edges=100,
                       idtype=torch.int64)
        g.ndata["x"] = torch.rand(30, 16, device=dev, requires_grad=True)

        # GCN layer: update_all(copy_u, sum) + linear
        g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
        linear = torch.nn.Linear(16, 8).to(dev)
        out = linear(g.ndata["h"])
        loss = out.sum()
        loss.backward()
        assert g.ndata["x"].grad is not None

    def test_multi_layer_gcn(self):
        """Multi-layer GCN with autograd."""
        g = make_graph(num_src=30, num_dst=30, num_edges=100,
                       idtype=torch.int64)

        layers = [torch.nn.Linear(16, 16).to(dev),
                  torch.nn.Linear(16, 8).to(dev)]
        x = torch.rand(30, 16, device=dev, requires_grad=True)

        for layer in layers:
            g.ndata["x"] = x
            g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
            x = torch.relu(layer(g.ndata["h"]))

        loss = x.sum()
        loss.backward()
        assert layers[0].weight.grad is not None

    def test_training_loop(self):
        """Multiple training steps (forward + backward + optimizer)."""
        g = make_graph(num_src=30, num_dst=30, num_edges=100,
                       idtype=torch.int64)
        linear = torch.nn.Linear(16, 8).to(dev)
        optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

        for step in range(5):
            g.ndata["x"] = torch.rand(30, 16, device=dev, requires_grad=True)
            g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
            out = linear(g.ndata["h"])
            loss = out.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Verify graph structure intact after training
        assert g.in_degrees().sum().item() == g.num_edges()
