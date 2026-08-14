"""
SDDMM operator tests on Ascend NPU.

Tests cover:
- dot/add/sub/mul/div/copy_lhs/copy_rhs × COO/CSR × FP32/FP16 × int32/int64
- Backward (autograd)
- Various feat_dims, bipartite, empty graph, large graph
"""
import pytest
import torch
import torch_npu
import numpy as np
import dgl
from dgl.ops import gsddmm

dev = torch.device("npu:0")
torch.npu.set_device(dev)


def make_graph(num_src=10, num_dst=8, num_edges=30, idtype=torch.int64):
    torch.npu.synchronize()
    np.random.seed(42)
    src = np.random.randint(0, num_src, num_edges)
    dst = np.random.randint(0, num_dst, num_edges)
    return dgl.graph((src.tolist(), dst.tolist()), idtype=idtype).to(dev)


def make_bipartite(num_src=10, num_dst=12, num_edges=40, idtype=torch.int64):
    np.random.seed(123)
    src = np.random.randint(0, num_src, num_edges)
    dst = np.random.randint(0, num_dst, num_edges)
    g = dgl.heterograph({("_U", "_E", "_V"): (src.tolist(), dst.tolist())})
    return g.astype(idtype).to(dev)


class TestSDDMM:
    """SDDMM operator tests."""

    def setup_method(self):
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
        if op == "div":
            rhs = torch.rand(g.num_src_nodes(), 4, device=dev) + 0.5
        else:
            rhs = torch.rand(g.num_src_nodes(), 4, device=dev)

        e_npu = gsddmm(g, op, lhs, rhs, lhs_target="u", rhs_target="v")
        e_cpu = gsddmm(g_cpu, op, lhs.cpu(), rhs.cpu(),
                        lhs_target="u", rhs_target="v")
        diff = (e_npu.cpu() - e_cpu).abs().max().item()
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
