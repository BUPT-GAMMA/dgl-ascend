"""Multi-relation Ascend CSR SDDMM tests with independent CPU references.

Run after sourcing CANN's set_env.sh with
``PYTHONPATH=python:$PYTHONPATH python3 tests/ascend/test_sddmm_csr_hetero_npu.py``.
The internal entry exercises all seven native operations; public API tests
also cover inference dispatch and the existing sub/div gradient rewrites.
"""

import itertools
import math
import unittest
from dataclasses import dataclass

import torch

from bench_sddmm_csr_npu import OPS, TARGETS, dgl, npu_available
from dgl._sparse_ops import _gsddmm_hetero


NODE_COUNTS = {"a": 10, "b": 7, "c": 6, "z": 9}
RELATIONS = (
    ("a", "r0", "z"),
    ("b", "r1", "a"),
    ("z", "r2", "c"),
)
ROW_COUNTS = (
    (0, 3, 0, 0, 1, 6, 2, 0, 4, 0),
    (1, 0, 4, 0, 2, 3, 0),
    (0, 2, 1, 0, 5, 0, 0, 1, 3),
)


@dataclass
class PreparedHetero:
    graph: dgl.DGLGraph
    targets: tuple
    lhs_cpu: list
    rhs_cpu: list
    lhs_npu: list
    rhs_npu: list
    rows: dict


def input_id(graph, target, etype):
    if target == "e":
        return graph.get_etype_id(etype)
    return graph.get_ntype_id(etype[0 if target == "u" else 2])


def prepare_hetero(device, targets=("u", "v"), layouts=None,
                   idtype=torch.int32, dtype=torch.float32, mapped=True,
                   empty_first=False, all_empty=False, default_format=False,
                   shared_source=False):
    """Retain original CSR rows/columns/EIDs for CPU-only reference gathers."""
    if layouts is None:
        layouts = [((7,), (7,))] * len(RELATIONS)
    relations = list(RELATIONS)
    row_counts = list(ROW_COUNTS)
    if shared_source:
        relations.append(("a", "r3", "c"))
        row_counts.append((1, 0, 0, 2, 0, 0, 1, 0, 3, 0))
        layouts = list(layouts) + [layouts[0]]
    generator = torch.Generator().manual_seed(2026)
    graph_data, rows = {}, {}
    for index, (etype, counts) in enumerate(zip(relations, row_counts)):
        counts = torch.tensor(counts, dtype=torch.int64)
        if all_empty or (empty_first and index == 0):
            counts.zero_()
        edge_count = int(counts.sum())
        source = torch.repeat_interleave(torch.arange(len(counts)), counts)
        indptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
        destination = torch.randint(NODE_COUNTS[etype[2]], (edge_count,),
                                    generator=generator)
        has_map = mapped if isinstance(mapped, bool) else mapped[index]
        eids = (torch.randperm(edge_count, generator=generator) if has_map else
                torch.arange(edge_count))
        rows[etype] = {"u": source, "v": destination, "e": eids}
        if default_format:
            source_eid, destination_eid = torch.empty_like(source), torch.empty_like(destination)
            source_eid[eids], destination_eid[eids] = source, destination
            graph_data[etype] = (source_eid.to(idtype), destination_eid.to(idtype))
        else:
            data = eids.to(idtype) if has_map else torch.empty(0, dtype=idtype)
            graph_data[etype] = ("csr", (indptr.to(idtype), destination.to(idtype), data))
    graph = dgl.heterograph(graph_data, num_nodes_dict=NODE_COUNTS, idtype=idtype)
    if not default_format:
        graph = graph.formats("csr")
    graph = graph.to(device)
    features = []
    for side, target in enumerate(targets):
        values = [None] * (len(graph.canonical_etypes) if target == "e" else len(graph.ntypes))
        for etype, layout in zip(relations, layouts):
            index = input_id(graph, target, etype)
            count = len(rows[etype]["e"]) if target == "e" else NODE_COUNTS[etype[0 if target == "u" else 2]]
            shape = (count,) + layout[side]
            if values[index] is not None:
                assert tuple(values[index].shape) == shape
                continue
            # Nonzero denominators make division errors informative.
            values[index] = torch.rand(shape, generator=generator, dtype=dtype) + 0.5
            if side == 0:
                values[index] -= 1.0
        features.append(values)
    lhs, rhs = features
    to_device = lambda values: [None if value is None else value.to(device) for value in values]
    return PreparedHetero(graph, targets, lhs, rhs, to_device(lhs), to_device(rhs), rows)


def run_hetero(prepared, op, public=False):
    lhs = None if op == "copy_rhs" else prepared.lhs_npu
    rhs = None if op == "copy_lhs" else prepared.rhs_npu
    if public:
        return dgl.ops.gsddmm(prepared.graph, op, lhs, rhs, *prepared.targets)
    # An unused side need not contain any placeholder tensors.
    lhs, rhs = () if lhs is None else tuple(lhs), () if rhs is None else tuple(rhs)
    return _gsddmm_hetero(prepared.graph._graph, op, len(lhs), *prepared.targets,
                         lhs_and_rhs_tuple=lhs + rhs)


def reference_outputs(prepared, op):
    expected = []
    for etype in prepared.graph.canonical_etypes:
        selected, scalar = [], []
        for side, (values, target) in enumerate(zip(
                (prepared.lhs_cpu, prepared.rhs_cpu), prepared.targets)):
            if (side == 0 and op == "copy_rhs") or (side == 1 and op == "copy_lhs"):
                selected.append(None)
                scalar.append(True)
                continue
            value = values[input_id(prepared.graph, target, etype)]
            scalar.append(value.ndim == 1)
            value = value[prepared.rows[etype][target]]
            selected.append(value.unsqueeze(-1) if value.ndim == 1 else value)
        lhs, rhs = selected
        if op == "copy_lhs":
            result = lhs
        elif op == "copy_rhs":
            result = rhs
        else:
            rank = max(lhs.ndim, rhs.ndim)
            lhs = lhs.reshape((len(lhs),) + (1,) * (rank - lhs.ndim) + tuple(lhs.shape[1:]))
            rhs = rhs.reshape((len(rhs),) + (1,) * (rank - rhs.ndim) + tuple(rhs.shape[1:]))
            result = ((lhs * rhs).sum(-1, keepdim=True) if op == "dot" else
                      {"add": torch.add, "sub": torch.sub,
                       "mul": torch.mul, "div": torch.div}[op](lhs, rhs))
        if all(scalar):
            result = result.squeeze(-1)
        ordered = torch.empty_like(result)
        ordered[prepared.rows[etype]["e"]] = result
        expected.append(ordered)
    return expected


@unittest.skipUnless(npu_available(), "Ascend NPU is required")
class TestSDDMMCsrHetero(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.npu.set_device(0)
        cls.device = torch.device("npu:0")
        cls.checked_cases = 0
        cls.checked_relations = 0

    @classmethod
    def tearDownClass(cls):
        torch.npu.synchronize(cls.device)
        print(f"Validated {cls.checked_cases} hetero CSR SDDMM cases "
              f"({cls.checked_relations} relation outputs)", flush=True)

    def check(self, prepared, op, public=False):
        actual = run_hetero(prepared, op, public)
        expected = reference_outputs(prepared, op)
        self.assertEqual(len(actual), len(expected))
        for result, reference in zip(actual, expected):
            tolerance = 1e-12 if reference.dtype == torch.float64 else 2e-5
            atol = tolerance
            if op == "dot":
                atol *= math.sqrt(max(value.shape[-1] if value.ndim > 1 else 1
                                     for value in prepared.lhs_cpu if value is not None))
            if op in ("copy_lhs", "copy_rhs"):
                tolerance = atol = 0
            torch.testing.assert_close(result.cpu(), reference, rtol=tolerance, atol=atol)
            type(self).checked_relations += 1
        type(self).checked_cases += 1

    def test_all_operations_targets_types_and_edge_maps(self):
        for idtype, dtype, targets, mapped in itertools.product(
                (torch.int32, torch.int64), (torch.float32, torch.float64),
                itertools.product(TARGETS, repeat=2),
                ((False, True, False), (True, False, True))):
            prepared = prepare_hetero(self.device, targets, idtype=idtype,
                                      dtype=dtype, mapped=mapped)
            for op in OPS:
                with self.subTest(op=op, targets=targets, idtype=idtype,
                                  dtype=dtype, mapped=mapped):
                    self.check(prepared, op)
            self.assertEqual(prepared.graph.formats()["created"], ["csr"])

    def test_relation_specific_broadcast_and_dot_widths(self):
        layouts = [((3, 1), (1, 5)), ((7,), (2, 1, 7)), ((2, 1, 9), (1, 4, 9))]
        dot_layouts = [((9,), (2, 9)), ((8, 16), (1, 16)), ((3, 1, 65), (1, 2, 65))]
        for idtype, dtype, targets in itertools.product(
                (torch.int32, torch.int64), (torch.float32, torch.float64),
                (("u", "v"), ("v", "u"), ("u", "e"), ("e", "v"), ("e", "e"))):
            for layout, ops in ((layouts, OPS[:-1]), (dot_layouts, ("dot",))):
                prepared = prepare_hetero(self.device, targets, layout,
                                          idtype=idtype, dtype=dtype)
                for op in ops:
                    with self.subTest(op=op, targets=targets, idtype=idtype, dtype=dtype):
                        self.check(prepared, op)

    def test_empty_relations_and_copy_missing_operands(self):
        for idtype, all_empty, targets in itertools.product(
                (torch.int32, torch.int64), (False, True),
                itertools.product(TARGETS, repeat=2)):
            prepared = prepare_hetero(self.device, targets, idtype=idtype,
                                      empty_first=True, all_empty=all_empty)
            for op in OPS:
                with self.subTest(op=op, targets=targets, all_empty=all_empty, idtype=idtype):
                    self.check(prepared, op)
                    if op in ("copy_lhs", "copy_rhs"):
                        self.check(prepared, op, public=True)

    def test_scalar_features_are_expanded_per_relation(self):
        layouts = [((), ()), ((1,), (1,)), ((3, 1), (1, 1))]
        for targets in (("u", "v"), ("e", "e"), ("v", "u")):
            prepared = prepare_hetero(self.device, targets, layouts)
            for op in OPS:
                with self.subTest(op=op, targets=targets):
                    self.check(prepared, op)
                    self.check(prepared, op, public=True)

    def test_shared_node_features(self):
        for targets in (("u", "v"), ("v", "u"), ("u", "u"), ("v", "v")):
            prepared = prepare_hetero(self.device, targets, shared_source=True)
            for op in OPS:
                with self.subTest(op=op, targets=targets):
                    self.check(prepared, op)

    def test_noncontiguous_inputs_through_dlpack(self):
        # DGL's existing tensor-to-NDArray conversion owns contiguity handling.
        for dtype, targets in itertools.product(
                (torch.float32, torch.float64), (("u", "v"), ("e", "e"))):
            prepared = prepare_hetero(
                self.device, targets, [((3, 17), (3, 17))] * len(RELATIONS),
                dtype=dtype)
            for values in (prepared.lhs_npu, prepared.rhs_npu):
                for index, value in enumerate(values):
                    if value is not None:
                        values[index] = value.transpose(-1, -2).contiguous().transpose(-1, -2)
                        self.assertFalse(values[index].is_contiguous())
            for op in OPS:
                with self.subTest(op=op, targets=targets, dtype=dtype):
                    self.check(prepared, op)
                    self.check(prepared, op, public=True)

    def test_npu_formats_normalized_before_dlpack(self):
        import torch_npu

        # This torch_npu version exposes a setter but no config attribute getter.
        old_internal_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
        torch.npu.config.allow_internal_format = True
        self.addCleanup(setattr, torch.npu.config, "allow_internal_format", old_internal_format)
        # DLPack does not describe torch_npu's private FRACTAL_NZ padding.
        # Normalize to ND while the Torch format descriptor is still available.
        # Raw NZ is outside this interface's input contract.
        for npu_format in (2, 29):  # ND, FRACTAL_NZ
            prepared = prepare_hetero(
                self.device, ("u", "v"), [((3, 17), (3, 17))] * len(RELATIONS))
            for values in (prepared.lhs_npu, prepared.rhs_npu):
                for index, value in enumerate(values):
                    if value is not None:
                        values[index] = torch_npu.npu_format_cast(value, npu_format)
                        self.assertEqual(torch_npu.get_npu_format(values[index]), npu_format)
                        if npu_format != 2:
                            values[index] = torch_npu.npu_format_cast(values[index], 2)
                        self.assertEqual(torch_npu.get_npu_format(values[index]), 2)
            for op in OPS:
                with self.subTest(op=op, npu_format=npu_format):
                    self.check(prepared, op)
                    self.check(prepared, op, public=True)

    def test_public_inference_and_default_csr_dispatch(self):
        for idtype, dtype, targets in itertools.product(
                (torch.int32, torch.int64), (torch.float32, torch.float64),
                (("u", "v"), ("v", "u"), ("e", "v"))):
            prepared = prepare_hetero(self.device, targets, idtype=idtype,
                                      dtype=dtype, default_format=True)
            self.assertEqual(prepared.graph.formats()["created"], ["coo"])
            for op in OPS:
                with self.subTest(op=op, targets=targets, idtype=idtype, dtype=dtype):
                    self.check(prepared, op, public=True)
                    self.assertIn("csr", prepared.graph.formats()["created"])
                    for value in prepared.lhs_npu + prepared.rhs_npu:
                        if value is not None:
                            value.requires_grad_(True)
                    with torch.no_grad():
                        self.check(prepared, op, public=True)
                    for value in prepared.lhs_npu + prepared.rhs_npu:
                        if value is not None:
                            value.requires_grad_(False)

    def test_public_sub_div_edge_gradients(self):
        # Edge-to-edge operands isolate SDDMM autograd from SpMMHetero backward.
        for op in ("sub", "div"):
            prepared = prepare_hetero(self.device, ("e", "e"),
                                      [((3, 1), (1, 5))] * len(RELATIONS))
            left = [value.detach().requires_grad_(True) for value in prepared.lhs_cpu]
            right = [value.detach().requires_grad_(True) for value in prepared.rhs_cpu]
            expected = [lhs - rhs if op == "sub" else lhs / rhs for lhs, rhs in zip(left, right)]
            weights = [torch.linspace(-1., 2., value.numel()).reshape(value.shape) for value in expected]
            for value in prepared.lhs_npu + prepared.rhs_npu:
                value.requires_grad_(True)
            expected_grads = torch.autograd.grad(expected, left + right, weights)
            actual = run_hetero(prepared, op, public=True)
            actual_grads = torch.autograd.grad(
                actual, prepared.lhs_npu + prepared.rhs_npu,
                [value.to(self.device) for value in weights])
            with self.subTest(op=op):
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result.cpu(), reference, rtol=2e-5, atol=2e-5)
                for result, reference in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(result.cpu(), reference, rtol=2e-5, atol=2e-5)
                type(self).checked_cases += 1
                type(self).checked_relations += len(actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
