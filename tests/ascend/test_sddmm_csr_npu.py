"""Native CSR SDDMM integration tests; run with PYTHONPATH=python python3 <file>.

The internal entry point ensures sub/div reach their native C++ interfaces.
References use original CSR rows, columns and edge IDs exclusively on CPU.
"""

import itertools
import math
import unittest

import torch

from bench_sddmm_csr_npu import (
    Case, OPS, TARGETS, check_accuracy, dgl, npu_available,
    prepare_case, run_operator,
)


@unittest.skipUnless(npu_available(), "Ascend NPU is required")
class TestSDDMMCsr(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.npu.set_device(0)
        cls.device = torch.device("npu:0")
        cls.checked_cases = 0

    @classmethod
    def tearDownClass(cls):
        torch.npu.synchronize(cls.device)
        print(f"Validated {cls.checked_cases} native CSR SDDMM cases", flush=True)

    def prepare(self, targets, shapes, idtype=torch.int32, mapped=False,
                dtype=torch.float32, counts=None):
        if counts is None:
            # Empty rows, repeated destinations, unsorted columns and arbitrary EIDs.
            counts = torch.tensor([0, 3, 0, 0, 1, 6, 2, 0, 4, 0])
        case = Case("unit", targets, (len(counts), 7, int(counts.sum())), *shapes)
        return prepare_case(case, self.device, idtype, dtype, mapped, counts=counts)

    def check(self, prepared, op, public=False):
        check_accuracy(prepared, op, run_operator(prepared, op, public=public))
        type(self).checked_cases += 1

    def test_binary_target_map_broadcast_matrix(self):
        shapes = (((1,), (1,)), ((7,), (7,)), ((9,), (9,)),
                  ((128,), (128,)), ((1,), (17,)), ((17,), (1,)),
                  ((3, 1), (1, 5)), ((5,), (2, 1, 5)))
        for idtype, mapped, targets, layout in itertools.product(
                (torch.int32, torch.int64), (False, True),
                itertools.product(TARGETS, repeat=2), shapes):
            prepared = self.prepare(targets, layout, idtype, mapped)
            for op in ("add", "sub", "mul", "div"):
                with self.subTest(op=op, idtype=idtype, mapped=mapped,
                                  targets=targets, shapes=layout):
                    self.check(prepared, op)

    def test_copy_missing_operand(self):
        for idtype, mapped, targets, shape in itertools.product(
                (torch.int32, torch.int64), (False, True),
                itertools.product(TARGETS, repeat=2), ((1,), (13,), (8, 16))):
            # Incompatible ignored operand shape must never affect the output.
            prepared = self.prepare(targets, (shape, (5, 7)), idtype, mapped)
            for op in ("copy_lhs", "copy_rhs"):
                with self.subTest(op=op, idtype=idtype, mapped=mapped,
                                  targets=targets, shape=shape):
                    self.check(prepared, op)

    def test_dot_last_dimension_and_broadcast(self):
        shapes = (((1,), (1,)), ((7,), (7,)), ((9,), (9,)),
                  ((128,), (128,)), ((8, 16), (8, 16)),
                  ((2, 128), (2, 128)), ((3, 72), (3, 72)),
                  ((3, 1, 13), (1, 5, 13)), ((13,), (2, 3, 13)))
        for idtype, mapped, targets, layout in itertools.product(
                (torch.int32, torch.int64), (False, True),
                itertools.product(TARGETS, repeat=2), shapes):
            with self.subTest(idtype=idtype, mapped=mapped, targets=targets, shapes=layout):
                self.check(self.prepare(targets, layout, idtype, mapped), "dot")

    def test_feature_and_reduction_chunks(self):
        for idtype, mapped, targets in itertools.product(
                (torch.int32, torch.int64), (False, True),
                (("u", "v"), ("v", "u"), ("u", "u"), ("e", "v"))):
            prepared = self.prepare(targets, ((32771,), (32771,)), idtype, mapped)
            for op in OPS:
                with self.subTest(op=op, idtype=idtype, mapped=mapped, targets=targets):
                    self.check(prepared, op)
            broadcast = self.prepare(targets, ((17, 1, 13), (1, 257, 13)), idtype, mapped)
            with self.subTest(op="dot", broadcast=True, idtype=idtype, mapped=mapped, targets=targets):
                self.check(broadcast, "dot")

    def test_dot_packed_output_boundaries(self):
        # Single-head outputs need 4B writes even when adjacent cores share a
        # 32B block. Large edge counts also cross multiple edge tiles per core.
        edge_counts = (1, 7, 8, 9, 63, 64, 65, 247, 248, 249, 255, 256, 257,
                       16383, 16384, 16385)
        layouts = ((("e", "v"), (64,)), (("v", "u"), (128,)),
                   (("u", "v"), (8, 16)))
        for edges, idtype, mapped, (targets, shape) in itertools.product(
                edge_counts, (torch.int32, torch.int64), (False, True), layouts):
            counts = torch.full((17,), edges // 17, dtype=torch.int64)
            counts[:edges % 17] += 1
            with self.subTest(edges=edges, idtype=idtype, mapped=mapped,
                              targets=targets, shape=shape):
                prepared = self.prepare(targets, (shape, shape), idtype, mapped,
                                        counts=counts)
                self.check(prepared, "dot")

    def test_dot_packed_reduction_layouts(self):
        # Vary the number of reduction chunks and leave an incomplete final
        # vector of packed heads, including output rows that are not 32B wide.
        shapes = ((8,), (32,), (72,), (256,), (1, 16), (3, 16), (9, 16))
        for edges, mapped, shape, targets in itertools.product(
                (257, 4097), (False, True), shapes,
                (("e", "v"), ("u", "v"))):
            counts = torch.full((17,), edges // 17, dtype=torch.int64)
            counts[:edges % 17] += 1
            with self.subTest(edges=edges, mapped=mapped, shape=shape,
                              targets=targets):
                prepared = self.prepare(targets, (shape, shape), torch.int64,
                                        mapped, counts=counts)
                self.check(prepared, "dot")

    def test_float64(self):
        for idtype, mapped, targets in itertools.product(
                (torch.int32, torch.int64), (False, True),
                (("u", "v"), ("v", "u"), ("e", "e"))):
            prepared = self.prepare(targets, ((3, 1, 13), (1, 2, 13)),
                                    idtype, mapped, dtype=torch.float64)
            for op in OPS:
                with self.subTest(op=op, idtype=idtype, mapped=mapped, targets=targets):
                    self.check(prepared, op)

    def test_float64_special_values(self):
        tiny = torch.finfo(torch.float64).tiny
        maximum = torch.finfo(torch.float64).max
        pairs = torch.tensor([
            (0., 0.), (-0., 0.), (0., -0.), (-0., -0.),
            (tiny, 5e-324), (5e-324, 2.), (tiny, math.nextafter(tiny, 0.)),
            (math.nextafter(1., 2.), 1.), (1., math.nextafter(1., 2.)),
            (maximum, maximum), (maximum, 2.), (1., 0.), (-1., 0.),
            (float("inf"), float("inf")), (float("inf"), -float("inf")),
            (float("nan"), 1.), (1., float("nan")),
        ], dtype=torch.float64)
        for idtype, mapped, targets in itertools.product(
                (torch.int32, torch.int64), (False, True), (("u", "v"), ("v", "u"))):
            prepared = self.prepare(targets, ((len(pairs),), (len(pairs),)),
                                    idtype, mapped, dtype=torch.float64)
            prepared.lhs_cpu = pairs[:, 0].repeat(len(prepared.lhs_cpu), 1)
            prepared.rhs_cpu = pairs[:, 1].repeat(len(prepared.rhs_cpu), 1)
            prepared.lhs_npu = prepared.lhs_cpu.to(self.device)
            prepared.rhs_npu = prepared.rhs_cpu.to(self.device)
            for op in ("sub", "mul", "div", "copy_lhs", "copy_rhs"):
                with self.subTest(op=op, idtype=idtype, mapped=mapped, targets=targets):
                    expected = (pairs[:, 0] if op == "copy_lhs" else
                                pairs[:, 1] if op == "copy_rhs" else
                                {"sub": torch.sub, "mul": torch.mul, "div": torch.div}[op](
                                    pairs[:, 0], pairs[:, 1]))
                    expected = expected.repeat(prepared.case.graph[2], 1)
                    actual = run_operator(prepared, op).cpu()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                    finite = ~torch.isnan(expected)
                    self.assertTrue(torch.equal(actual.view(torch.int64)[finite],
                                                expected.view(torch.int64)[finite]))
                    type(self).checked_cases += 1

    def test_float64_dot_reduction_chunks(self):
        prepared = self.prepare(("u", "v"), ((32771,), (32771,)),
                                torch.int64, mapped=True, dtype=torch.float64)
        self.check(prepared, "dot")

    def test_dot_width_one_signed_zero(self):
        for dtype in (torch.float32, torch.float64):
            prepared = self.prepare(("e", "e"), ((1,), (1,)), mapped=True, dtype=dtype)
            pairs = torch.tensor([[-0., 1.], [0., -1.], [-2., 0.],
                                  [2., -3.], [-2., -3.], [0., 1.],
                                  [1., -0.], [-1., -0.]], dtype=dtype).repeat(2, 1)
            prepared.lhs_npu = pairs[:, :1].contiguous().to(self.device)
            prepared.rhs_npu = pairs[:, 1:].contiguous().to(self.device)
            with self.subTest(dtype=dtype):
                expected = (pairs[:, :1] * pairs[:, 1:]).sum(-1, keepdim=True)
                actual = run_operator(prepared, "dot").cpu()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(torch.signbit(actual), torch.signbit(expected)))
                type(self).checked_cases += 1

    def test_empty_edges(self):
        for idtype, targets, counts in itertools.product(
                (torch.int32, torch.int64), (("u", "v"), ("e", "u")),
                (torch.zeros(10, dtype=torch.int64), torch.empty(0, dtype=torch.int64))):
            prepared = self.prepare(targets, ((8, 16), (8, 16)), idtype, counts=counts)
            for op in OPS:
                with self.subTest(op=op, idtype=idtype, targets=targets, rows=len(counts)):
                    self.check(prepared, op)

    def test_index_and_row_cache_refills(self):
        counts = torch.ones(600, dtype=torch.int64)
        counts[0] = 9000
        counts[1:280] = 0
        counts[300] = 800000
        for idtype, mapped, targets in itertools.product(
                (torch.int32, torch.int64), (False, True), (("u", "v"), ("v", "u"))):
            prepared = self.prepare(targets, ((1,), (1,)), idtype, mapped, counts=counts)
            for op in ("sub", "div", "copy_lhs", "copy_rhs", "dot"):
                with self.subTest(op=op, idtype=idtype, mapped=mapped, targets=targets):
                    self.check(prepared, op)

    def test_public_api_inference(self):
        for targets in (("u", "v"), ("v", "u"), ("u", "e")):
            prepared = self.prepare(targets, ((8, 16), (8, 16)), mapped=True)
            for op in OPS:
                with self.subTest(op=op, targets=targets, mode="requires_grad_false"):
                    self.check(prepared, op, public=True)
                prepared.lhs_npu.requires_grad_(True)
                prepared.rhs_npu.requires_grad_(True)
                with torch.no_grad(), self.subTest(op=op, targets=targets, mode="no_grad"):
                    self.check(prepared, op, public=True)
                prepared.lhs_npu.requires_grad_(False)
                prepared.rhs_npu.requires_grad_(False)

    def test_public_sub_div_gradients(self):
        # Edge inputs exercise both gradients without relying on SpMM backward.
        # The public backend retains differentiable negation/reciprocal rewrites.
        for op, mapped in itertools.product(("sub", "div"), (False, True)):
            prepared = self.prepare(("e", "e"), ((3, 1), (1, 5)), mapped=mapped)
            prepared.lhs_npu.requires_grad_(True)
            prepared.rhs_npu.requires_grad_(True)
            left = prepared.lhs_cpu.detach().requires_grad_(True)
            right = prepared.rhs_cpu.detach().requires_grad_(True)
            with self.subTest(op=op, mapped=mapped):
                expected = left - right if op == "sub" else left / right
                weights = torch.linspace(-1., 2., expected.numel()).reshape(expected.shape)
                expected_grads = torch.autograd.grad(expected, (left, right), weights)
                actual = run_operator(prepared, op, public=True)
                actual_grads = torch.autograd.grad(
                    actual, (prepared.lhs_npu, prepared.rhs_npu), weights.to(self.device))
                torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-5)
                for result, reference in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(result.cpu(), reference, rtol=2e-5, atol=2e-5)
                type(self).checked_cases += 1

    def test_default_graph_prefers_csr(self):
        for op in OPS:
            graph = dgl.graph(([2, 0, 2, 1], [0, 1, 1, 2]), num_nodes=3).to(self.device)
            lhs = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(self.device)
            rhs = (lhs + 1).to(self.device)
            with self.subTest(op=op):
                actual = dgl.ops.gsddmm(graph, op, None if op == "copy_rhs" else lhs,
                                       None if op == "copy_lhs" else rhs).cpu()
                left = lhs.cpu()[[2, 0, 2, 1]]
                right = rhs.cpu()[[0, 1, 1, 2]]
                expected = (left if op == "copy_lhs" else right if op == "copy_rhs" else
                            (left * right).sum(-1, keepdim=True) if op == "dot" else
                            {"add": torch.add, "sub": torch.sub,
                             "mul": torch.mul, "div": torch.div}[op](left, right))
                torch.testing.assert_close(actual, expected)
                self.assertIn("csr", graph.formats()["created"])
                type(self).checked_cases += 1


if __name__ == "__main__":
    unittest.main(verbosity=2)
