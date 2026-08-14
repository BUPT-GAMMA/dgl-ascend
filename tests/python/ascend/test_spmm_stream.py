"""
SPMM operator and stream sync tests on Ascend NPU.

Tests cover:
- SPMM: copy_lhs + sum/max/min × FP32 × int32/int64
- update_all(copy_u, sum/max/min) fused path
- UDF reduce (degree bucketing path)
- Backward (autograd)
- Stream sync: in_edges/in_degrees after update_all/gsddmm
- segment_reduce (mean_nodes)
- End-to-end: GCN forward/backward, multi-layer, training loop
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


def make_graph(num_src=10, num_dst=8, num_edges=30, idtype=torch.int64):
    torch.npu.synchronize()
    np.random.seed(42)
    src = np.random.randint(0, num_src, num_edges)
    dst = np.random.randint(0, num_dst, num_edges)
    return dgl.graph((src.tolist(), dst.tolist()), idtype=idtype).to(dev)


class TestSPMM:
    """SPMM operator tests."""

    def setup_method(self):
        torch.npu.synchronize()

    @pytest.mark.parametrize("reducer", ["sum", "max", "min"])
    @pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
    def test_spmm_copy_lhs_sum(self, reducer, idtype):
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
        g = make_graph(idtype=torch.int64)
        ufeat = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)
        v = gspmm(g, "copy_lhs", "sum", ufeat, None)
        loss = v.sum()
        loss.backward()
        assert ufeat.grad is not None

    @pytest.mark.parametrize("feat_dim", [1, 4, 13, 64, 128])
    def test_spmm_feat_dims(self, feat_dim):
        g = make_graph(idtype=torch.int64)
        g_cpu = g.cpu()
        ufeat = torch.rand(g.num_src_nodes(), feat_dim, device=dev)

        v_npu = gspmm(g, "copy_lhs", "sum", ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", "sum", ufeat.cpu(), None)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-3, f"feat_dim={feat_dim} diff={diff}"

    def test_spmm_large_graph(self):
        g = make_graph(num_src=1000, num_dst=1000, num_edges=5000,
                       idtype=torch.int64)
        g_cpu = g.cpu()
        ufeat = torch.rand(1000, 64, device=dev)

        v_npu = gspmm(g, "copy_lhs", "sum", ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", "sum", ufeat.cpu(), None)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-3, f"large graph diff={diff}"

    def test_spmm_zero_degree_nodes(self):
        g = dgl.graph(([1, 2, 3], [1, 2, 3]), idtype=torch.int64).to(dev)
        g_cpu = g.cpu()
        ufeat = torch.rand(4, 4, device=dev)

        v_npu = gspmm(g, "copy_lhs", "sum", ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", "sum", ufeat.cpu(), None)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4
        assert v_npu[0].abs().max() == 0


class TestStreamSync:
    """Stream synchronization and UDF reduce tests."""

    def setup_method(self):
        torch.npu.synchronize()

    def test_udf_reduce_forward(self):
        g = make_graph(idtype=torch.int64)
        g_cpu = g.cpu()
        g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev)
        g_cpu.ndata["x"] = g.ndata["x"].cpu()

        g.update_all(fn.copy_u("x", "m"),
                      lambda n: {"h": n.mailbox["m"].sum(1)})
        g_cpu.update_all(fn.copy_u("x", "m"),
                          lambda n: {"h": n.mailbox["m"].sum(1)})
        diff = (g.ndata["h"].cpu() - g_cpu.ndata["h"]).abs().max().item()
        assert diff < 1e-4

    def test_udf_reduce_backward(self):
        g = make_graph(idtype=torch.int64)
        g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev,
                                   requires_grad=True)
        g.update_all(fn.copy_u("x", "m"),
                      lambda n: {"h": n.mailbox["m"].sum(1)})
        loss = g.ndata["h"].sum()
        loss.backward()
        assert g.ndata["x"].grad is not None

    def test_in_edges_after_update_all(self):
        g = make_graph(num_src=20, num_dst=20, num_edges=100,
                       idtype=torch.int64)
        g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev,
                                   requires_grad=True)
        g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))

        nodes = torch.arange(g.num_nodes(), device=dev)
        eid = g.in_edges(nodes, form="eid")
        assert len(eid) == g.num_edges()
        assert g.in_degrees().sum().item() == g.num_edges()

    def test_in_edges_after_gsddmm(self):
        g = make_graph(num_src=20, num_dst=20, num_edges=100,
                       idtype=torch.int64)
        lhs = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)
        rhs = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)
        gsddmm(g, "dot", lhs, rhs, lhs_target="u", rhs_target="v")

        nodes = torch.arange(g.num_nodes(), device=dev)
        eid = g.in_edges(nodes, form="eid")
        assert len(eid) == g.num_edges()

    def test_repeated_update_all(self):
        g = make_graph(num_src=20, num_dst=20, num_edges=100,
                       idtype=torch.int64)
        for _ in range(3):
            g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev,
                                       requires_grad=True)
            g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
            g.ndata["h"].sum().backward()

        nodes = torch.arange(g.num_nodes(), device=dev)
        eid = g.in_edges(nodes, form="eid")
        assert len(eid) == g.num_edges()

    def test_segment_reduce(self):
        from dgl.ops import segment_reduce
        seglen = torch.tensor([3, 2, 1], device=dev)
        value = torch.rand(6, 4, device=dev, requires_grad=True)
        y = segment_reduce(seglen, value, reducer="sum")
        assert y.shape == (3, 4)
        y.sum().backward()
        assert value.grad is not None


class TestEndToEnd:
    """End-to-end GNN model tests."""

    def setup_method(self):
        torch.npu.synchronize()

    def test_gcn_forward_backward(self):
        g = make_graph(num_src=30, num_dst=30, num_edges=100,
                       idtype=torch.int64)
        g.ndata["x"] = torch.rand(30, 16, device=dev, requires_grad=True)
        g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
        linear = torch.nn.Linear(16, 8).to(dev)
        out = linear(g.ndata["h"])
        loss = out.sum()
        loss.backward()
        assert g.ndata["x"].grad is not None

    def test_multi_layer_gcn(self):
        g = make_graph(num_src=30, num_dst=30, num_edges=100,
                       idtype=torch.int64)
        layers = [torch.nn.Linear(16, 16).to(dev),
                  torch.nn.Linear(16, 8).to(dev)]
        x = torch.rand(30, 16, device=dev, requires_grad=True)
        for layer in layers:
            g.ndata["x"] = x
            g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
            x = torch.relu(layer(g.ndata["h"]))
        x.sum().backward()
        assert layers[0].weight.grad is not None

    def test_training_loop(self):
        g = make_graph(num_src=30, num_dst=30, num_edges=100,
                       idtype=torch.int64)
        linear = torch.nn.Linear(16, 8).to(dev)
        optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
        for _ in range(5):
            g.ndata["x"] = torch.rand(30, 16, device=dev, requires_grad=True)
            g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
            out = linear(g.ndata["h"])
            loss = out.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        assert g.in_degrees().sum().item() == g.num_edges()
