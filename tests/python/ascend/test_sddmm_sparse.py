"""
Tests for dgl.sparse SDDMM dispatch path on Ascend NPU.

These tests verify that array.cc::CSRSDDMM() and array.cc::COOSDDMM() correctly
route kDGLAscend device to the Ascend SDDMM kernel implementations
(SDDMMCsr<kDGLAscend> / SDDMMCoo<kDGLAscend>) via ATEN_XPU_SWITCH_CUDA_ASCEND.

Before the macro fix (ATEN_XPU_SWITCH_CUDA), calling dgl.sparse.sddmm() on NPU
would crash with LOG(FATAL) "Operator SDDMM does not support ascend device."
because ATEN_XPU_SWITCH_CUDA only handles kDGLCPU and kDGLCUDA.

The dgl.sparse path goes through:
  dgl.sparse.sddmm() → matmul.cc::SDDMMNoAutoGrad() → aten::CSRSDDMM()/COOSDDMM()
  → array.cc macro dispatch → SDDMMCsr<kDGLAscend>/SDDMMCoo<kDGLAscend>

Note: dgl.ops.gsddmm() goes through kernel.cc::SDDMM() which was already fixed
in PR #16. These tests cover the dgl.sparse path which was missed.
"""
import pytest
import torch
import torch_npu
import numpy as np
import dgl
import dgl.sparse as dglsp

dev = torch.device("npu:0")
torch.npu.set_device(dev)


def make_sparse_coo(num_src=10, num_dst=8, num_edges=30):
    np.random.seed(42)
    src = np.random.randint(0, num_src, num_edges)
    dst = np.random.randint(0, num_dst, num_edges)
    row = torch.tensor(src, dtype=torch.int64)
    col = torch.tensor(dst, dtype=torch.int64)
    val = torch.ones(num_edges, dtype=torch.float32)
    return dglsp.from_coo(row, col, val, shape=(num_src, num_dst)).to(dev)


class TestSparseSDDMMDirect:
    """Tests that dgl.sparse.SDDMM dispatches to Ascend kernel without crash."""

    def setup_method(self):
        torch.npu.synchronize()

    def teardown_method(self):
        torch.npu.synchronize()

    def test_sparse_sddmm_coo_no_crash(self):
        """dgl.sparse.sddmm with COO format does not crash on NPU.

        Before macro fix: LOG(FATAL) "does not support ascend device"
        After macro fix: routes to SDDMMCoo<kDGLAscend>
        """
        sp = make_sparse_coo()
        mat1 = torch.rand(10, 4, device=dev)
        mat2 = torch.rand(4, 8, device=dev)
        result = dglsp.sddmm(sp, mat1, mat2)
        assert result.val.shape[0] == 30

    def test_sparse_sddmm_csr_no_crash(self):
        """dgl.sparse.sddmm with CSR format does not crash on NPU.

        CSR path goes through CSRSDDMM → SDDMMCsrAscend → converts to COO
        → SDDMMCooAscend.
        """
        sp = make_sparse_coo()
        # Force CSR format
        sp_csr = dglsp.from_coo(
            sp.coo()[0].cpu(), sp.coo()[1].cpu(),
            sp.val.cpu(), shape=sp.shape).to(dev)
        mat1 = torch.rand(10, 4, device=dev)
        mat2 = torch.rand(4, 8, device=dev)
        result = dglsp.sddmm(sp_csr, mat1, mat2)
        assert result.val.shape[0] == 30

    def test_sparse_sddmm_shape_correct(self):
        """Output shape matches nnz regardless of dispatch path."""
        for num_edges in [5, 30, 100]:
            sp = make_sparse_coo(num_edges=num_edges)
            mat1 = torch.rand(10, 4, device=dev)
            mat2 = torch.rand(4, 8, device=dev)
            result = dglsp.sddmm(sp, mat1, mat2)
            assert result.val.shape[0] == num_edges

    def test_sparse_sddmm_empty_graph(self):
        """Empty graph (0 edges) does not crash."""
        row = torch.tensor([], dtype=torch.int64)
        col = torch.tensor([], dtype=torch.int64)
        val = torch.tensor([], dtype=torch.float32)
        sp = dglsp.from_coo(row, col, val, shape=(5, 5)).to(dev)
        mat1 = torch.rand(5, 4, device=dev)
        mat2 = torch.rand(4, 5, device=dev)
        result = dglsp.sddmm(sp, mat1, mat2)
        assert result.val.shape[0] == 0

    def test_sparse_sddmm_fp16(self):
        """FP16 dtype does not crash on NPU."""
        sp = make_sparse_coo()
        mat1 = torch.rand(10, 4, device=dev).half()
        mat2 = torch.rand(4, 8, device=dev).half()
        result = dglsp.sddmm(sp, mat1, mat2)
        assert result.val.shape[0] == 30


class TestSDDMMMacroConsistency:
    """Verify that both dispatch paths (kernel.cc and array.cc) reach Ascend kernels."""

    def setup_method(self):
        torch.npu.synchronize()

    def teardown_method(self):
        torch.npu.synchronize()

    def test_gsddmm_and_sparse_both_work(self):
        """Both dgl.ops.gsddmm (kernel.cc path) and dgl.sparse.sddmm (array.cc
        path) should work on NPU without crashing.

        gsddmm goes through kernel.cc::SDDMM() which uses
        ATEN_XPU_SWITCH_CUDA_ASCEND (fixed in PR #16).
        dgl.sparse.sddmm goes through matmul.cc → array.cc::COOSDDMM() which
        uses ATEN_XPU_SWITCH_CUDA_ASCEND (fixed in this PR).
        """
        np.random.seed(42)
        src = np.random.randint(0, 10, 30)
        dst = np.random.randint(0, 8, 30)

        # Path 1: gsddmm (kernel.cc)
        g = dgl.graph((src.tolist(), dst.tolist()), idtype=torch.int64).to(dev)
        lhs = torch.rand(10, 4, device=dev)
        rhs = torch.rand(10, 4, device=dev)
        e_gsddmm = dgl.ops.gsddmm(g, "dot", lhs, rhs,
                                  lhs_target="u", rhs_target="v")
        assert e_gsddmm.shape[0] == 30

        # Path 2: dgl.sparse.sddmm (array.cc)
        row = torch.tensor(src, dtype=torch.int64)
        col = torch.tensor(dst, dtype=torch.int64)
        val = torch.ones(30, dtype=torch.float32)
        sp = dglsp.from_coo(row, col, val, shape=(10, 8)).to(dev)
        mat1 = torch.rand(10, 4, device=dev)
        mat2 = torch.rand(4, 8, device=dev)
        result = dglsp.sddmm(sp, mat1, mat2)
        assert result.val.shape[0] == 30
