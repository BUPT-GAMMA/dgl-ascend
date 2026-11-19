"""Native CSR SDDMM add integration tests.

Run after configuring and building DGL with USE_ASCEND=ON:
    PYTHONPATH=python python3 tests/ascend/test_sddmm_csr_add_npu.py

All references are computed on CPU from the original CSR arrays. CSR-only
graphs ensure the test cannot silently exercise the COO fallback.
"""

import itertools
import math
import unittest

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

# Load torch_npu before libdgl.so.
import dgl


TARGETS = ("u", "e", "v")
ID_TYPES = (torch.int32, torch.int64)
FEATURE_TYPES = (torch.float32, torch.float64)
ETYPE = ("source", "edge", "destination")


def npu_available():
    return torch_npu is not None and torch.npu.is_available()


@unittest.skipUnless(npu_available(), "Ascend NPU is required")
class TestSDDMMCsrAdd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        cls.device = torch.device("npu:0")
        cls.checked_cases = 0

    @classmethod
    def tearDownClass(cls):
        torch.npu.synchronize()
        print(f"Validated {cls.checked_cases} CSR add cases", flush=True)

    def make_graph(self, idtype, mapped, counts=None):
        if counts is None:
            # Leading, trailing, and consecutive empty rows; unsorted columns.
            counts = torch.tensor([0, 3, 0, 0, 1, 6, 2, 0, 4, 0])
        indptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
        edges = int(indptr[-1])
        indices = (torch.arange(edges) * 5 + 3) % 7
        edge_ids = torch.arange(edges).flip(0) if mapped else torch.arange(edges)
        data = edge_ids.to(idtype) if mapped else torch.empty(0, dtype=idtype)
        graph = dgl.heterograph(
            {ETYPE: ("csr", (indptr.to(idtype), indices.to(idtype), data))},
            num_nodes_dict={"source": len(counts), "destination": 7},
            idtype=idtype,
        ).formats("csr").to(self.device)
        rows = torch.repeat_interleave(torch.arange(len(counts)), counts)
        return graph, {"u": rows, "e": edge_ids, "v": indices}, edge_ids

    def check_case(self, graph, row_indices, edge_ids, dtype, targets, shapes):
        sizes = {"u": graph.num_nodes("source"), "e": graph.num_edges(),
                 "v": graph.num_nodes("destination")}
        inputs = []
        for slot, (target, shape) in enumerate(zip(targets, shapes)):
            size = sizes[target] * math.prod(shape)
            values = torch.arange(size, dtype=dtype).reshape((sizes[target],) + shape)
            # Dyadic fractions are exactly representable in both dtypes.
            inputs.append(((values % 127) - 63) / (8 if slot == 0 else 16))
        lhs, rhs = inputs
        selected = [value[row_indices[target]]
                    for value, target in zip(inputs, targets)]
        rank = max(value.ndim for value in selected)
        selected = [value.reshape((len(edge_ids),) + (1,) * (rank - value.ndim)
                                  + tuple(value.shape[1:])) for value in selected]
        sums = selected[0] + selected[1]
        expected = torch.empty_like(sums)
        expected[edge_ids] = sums
        actual = dgl.ops.gsddmm(
            graph, "add", lhs.to(self.device), rhs.to(self.device),
            lhs_target=targets[0], rhs_target=targets[1],
        ).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(graph.formats()["created"], ["csr"])
        type(self).checked_cases += 1

    def test_target_dtype_map_broadcast_matrix(self):
        layouts = (
            ((1,), (1,)), ((13,), (13,)), ((64,), (64,)),
            ((128,), (128,)), ((1,), (17,)), ((17,), (1,)),
            ((3, 1), (1, 5)), ((5,), (2, 1, 5)),
        )
        for idtype, mapped in itertools.product(ID_TYPES, (False, True)):
            graph, rows, edge_ids = self.make_graph(idtype, mapped)
            for dtype, lhs_target, rhs_target, shapes in itertools.product(
                    FEATURE_TYPES, TARGETS, TARGETS, layouts):
                with self.subTest(idtype=idtype, mapped=mapped, dtype=dtype,
                                  targets=(lhs_target, rhs_target), shapes=shapes):
                    self.check_case(graph, rows, edge_ids, dtype,
                                    (lhs_target, rhs_target), shapes)

    def test_feature_alignment_boundaries(self):
        # Exercise real row lengths immediately below and above a 32B block.
        # Padding must stay in UB rather than extending the GM row stride.
        for idtype, mapped, dtype in itertools.product(
                ID_TYPES, (False, True), FEATURE_TYPES):
            graph, rows, edge_ids = self.make_graph(idtype, mapped)
            block_elements = 32 // torch.empty((), dtype=dtype).element_size()
            widths = (1, block_elements - 1, block_elements, block_elements + 1,
                      2 * block_elements - 1, 2 * block_elements,
                      2 * block_elements + 1)
            for width, targets in itertools.product(
                    widths, (("u", "v"), ("v", "u"), ("u", "u"), ("e", "v"))):
                with self.subTest(idtype=idtype, mapped=mapped, dtype=dtype,
                                  width=width, targets=targets):
                    self.check_case(graph, rows, edge_ids, dtype,
                                    targets, ((width,), (width,)))

    def test_feature_chunks(self):
        # Both layouts exceed UB capacity and require several feature chunks.
        for idtype, mapped, dtype, targets, shapes in itertools.product(
                ID_TYPES, (False, True), FEATURE_TYPES,
                (("u", "v"), ("v", "u"), ("u", "u"), ("e", "v")),
                (((32771,), (32771,)), ((17, 1), (1, 257)))):
            with self.subTest(idtype=idtype, mapped=mapped, dtype=dtype,
                              targets=targets, shapes=shapes):
                graph, rows, edge_ids = self.make_graph(idtype, mapped)
                self.check_case(graph, rows, edge_ids, dtype, targets, shapes)

    def test_index_and_row_cache_refills(self):
        counts = torch.ones(600, dtype=torch.int64)
        counts[0] = 9000
        counts[1:280] = 0
        # Each core receives enough edges to refill its index cache; u/v also
        # crosses 256-entry indptr pages after traversing the high-degree row.
        counts[300] = 800000
        for idtype, mapped in itertools.product(ID_TYPES, (False, True)):
            graph, rows, edge_ids = self.make_graph(idtype, mapped, counts)
            for targets in (("u", "v"), ("e", "v")):
                with self.subTest(idtype=idtype, mapped=mapped, targets=targets):
                    self.check_case(graph, rows, edge_ids, torch.float32,
                                    targets, ((1,), (1,)))

    def test_large_feature_shapes(self):
        # The four large layouts in the standalone operator's validation.
        cases = (
            (65536, ("e", "v"), (64,)),
            (200000, ("v", "u"), (128,)),
            (500000, ("u", "e"), (64,)),
            (500000, ("u", "v"), (8, 16)),
        )
        for edges, targets, shape in cases:
            with self.subTest(edges=edges, targets=targets, shape=shape):
                counts = torch.full((1024,), edges // 1024, dtype=torch.int64)
                counts[:edges % 1024] += 1
                graph, rows, edge_ids = self.make_graph(torch.int32, True, counts)
                self.check_case(graph, rows, edge_ids, torch.float32,
                                targets, (shape, shape))

    def test_empty_edges(self):
        for idtype, dtype in itertools.product(ID_TYPES, FEATURE_TYPES):
            for counts, shapes in (
                    (torch.zeros(10, dtype=torch.int64), ((13,), (13,))),
                    (torch.empty(0, dtype=torch.int64), ((13,), (13,)))):
                with self.subTest(idtype=idtype, dtype=dtype, shapes=shapes):
                    graph, rows, edge_ids = self.make_graph(idtype, False, counts)
                    self.check_case(graph, rows, edge_ids, dtype, ("u", "v"), shapes)

    def test_zero_features_follow_framework_validation(self):
        # DGL's common FFI contiguity check rejects zero-width PyTorch tensors
        # before dispatch on both CPU and Ascend; the CSR kernel is not called.
        graph, _, _ = self.make_graph(torch.int32, False)
        for device in (torch.device("cpu"), self.device):
            with self.subTest(device=device):
                with self.assertRaisesRegex(dgl.DGLError, "contiguous"):
                    dgl.ops.gsddmm(
                        graph.to(device), "add",
                        torch.empty(10, 0, 3, device=device),
                        torch.empty(7, 0, 3, device=device))

    def test_fp64_special_values(self):
        tiny = torch.finfo(torch.float64).tiny
        lhs = torch.tensor([0., -0., -0., tiny, 5e-324, 1., float("inf"),
                            -float("inf"), float("nan"), 1., float("inf")],
                           dtype=torch.float64)
        rhs = torch.tensor([0., -0., 0., -tiny, 5e-324, 2. ** -53, 1.,
                            float("inf"), 1., float("nan"), float("nan")],
                           dtype=torch.float64)
        next_one = math.nextafter(1., 2.)
        below_tiny = math.nextafter(tiny, 0.)
        maximum = torch.finfo(torch.float64).max
        boundary_pairs = torch.tensor([
            # Cancellation requires large normalization shifts in both orders.
            (next_one, -1.), (1., -next_one),
            (1., -math.nextafter(1., 0.)),
            # Subnormal results and the transition to the smallest normal value.
            (tiny, -below_tiny), (below_tiny, -tiny),
            (below_tiny, 5e-324), (tiny, -5e-324),
            # Halfway rounding with even and odd low bits, and a significand carry.
            (1., 2. ** -53), (next_one, 2. ** -53),
            (math.nextafter(2., 0.), 2. ** -53),
            (1., 2. ** -54),
            # Overflow and exact cancellation at the largest finite magnitude.
            (maximum, maximum), (-maximum, -maximum), (maximum, -maximum),
        ], dtype=torch.float64)
        lhs = torch.cat((lhs, boundary_pairs[:, 0]))
        rhs = torch.cat((rhs, boundary_pairs[:, 1]))
        for idtype, mapped, targets in itertools.product(
                ID_TYPES, (False, True), (("u", "v"), ("v", "u"), ("e", "e"))):
            graph, rows, edge_ids = self.make_graph(idtype, mapped)
            sizes = {"u": 10, "e": graph.num_edges(), "v": 7}
            left = lhs.repeat(sizes[targets[0]], 1)
            right = rhs.repeat(sizes[targets[1]], 1)
            actual = dgl.ops.gsddmm(
                graph, "add", left.to(self.device), right.to(self.device),
                lhs_target=targets[0], rhs_target=targets[1]).cpu()
            expected = (lhs + rhs).repeat(graph.num_edges(), 1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            finite = ~torch.isnan(expected)
            self.assertTrue(torch.equal(actual.view(torch.int64)[finite],
                                        expected.view(torch.int64)[finite]))
            type(self).checked_cases += 1

    def test_default_graph_prefers_available_csr(self):
        lhs = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        rhs = torch.arange(9, dtype=torch.float32).reshape(3, 3) / 2
        for formats in ("default", "all", "coo"):
            with self.subTest(formats=formats):
                graph = dgl.graph(([2, 0, 2, 1], [0, 1, 1, 2]), num_nodes=3)
                if formats == "all":
                    graph.create_formats_()
                elif formats == "coo":
                    graph = graph.formats("coo")
                graph = graph.to(self.device)
                actual = dgl.ops.u_add_v(
                    graph, lhs.to(self.device), rhs.to(self.device)).cpu()
                torch.testing.assert_close(actual, lhs[[2, 0, 2, 1]] + rhs[[0, 1, 1, 2]])
                if formats != "coo":
                    self.assertIn("csr", graph.formats()["created"])
                type(self).checked_cases += 1


if __name__ == "__main__":
    unittest.main(verbosity=2)
