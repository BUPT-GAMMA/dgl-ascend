#!/usr/bin/env python3
"""Compare DGL's native CPU and Ascend implementations of COOGetDataAndIndices.

Every case is constructed once as a CPU COO-only graph and then copied to the
NPU.  The test invokes the same DGL ``UnitGraph::EdgeIdsAll`` entry on both
devices, compares the complete ``(row, col, eid)`` result multiset, and reports
end-to-end operator latency for both implementations.

Run the four standard NNZ scales:

    PYTHONPATH=python python \
        tests/ascend/test_coo_get_data_and_indices_npu.py

Run both int32 and int64, or include the three optional large workloads:

    PYTHONPATH=python python \
        tests/ascend/test_coo_get_data_and_indices_npu.py --dtype both
    PYTHONPATH=python python \
        tests/ascend/test_coo_get_data_and_indices_npu.py --suite all \
        --warmup 1 --repeat 1 --batches 3

Graph construction, CPU-to-NPU input copies, and correctness copies are outside
the timed regions.  NPU timing includes the complete DGL operator path: Match,
count synchronization/prefix sum, output allocation, and Gather.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, Tuple

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

# DGL-Ascend requires torch and torch_npu to be loaded before libdgl.so.
import dgl  # noqa: E402


STANDARD_NNZ: Tuple[int, ...] = (1_000, 10_000, 50_000, 100_000)
DTYPES = {"int32": torch.int32, "int64": torch.int64}
DEFAULT_QUERY_COUNT = 128
DEFAULT_MATCHES_PER_QUERY = 8
DEFAULT_NUM_ROWS = 4_096
DEFAULT_NUM_COLS = 4_096


@dataclass(frozen=True)
class WorkloadCase:
    name: str
    num_rows: int
    num_cols: int
    nnz: int
    query_count: int
    hit_query_count: int
    matches_per_hit: int
    row_pattern: str = "fixed_degree"
    col_pattern: str = "fixed_degree"

    @property
    def expected_outputs(self) -> int:
        return min(self.nnz, self.hit_query_count * self.matches_per_hit)


@dataclass(frozen=True)
class Timing:
    median_ms: float
    minimum_ms: float
    maximum_ms: float
    samples_ms: Tuple[float, ...]


@dataclass(frozen=True)
class Comparison:
    case: str
    dtype: str
    nnz: int
    queries: int
    outputs: int
    cpu_ms: float
    npu_ms: float
    cpu_min_ms: float
    cpu_max_ms: float
    npu_min_ms: float
    npu_max_ms: float
    cpu_over_npu: float


LARGE_CASES: Tuple[WorkloadCase, ...] = (
    WorkloadCase(
        "typical", 10_000, 10_000, 500_000, 65_536, 32_768, 1,
        "balanced", "mixed"
    ),
    WorkloadCase(
        "unsorted", 4_096, 4_096, 65_536, 32_768, 32_768, 1,
        "balanced", "descending"
    ),
    WorkloadCase(
        "sampled_subgraph", 12_000, 2_000, 60_000, 60_000, 60_000, 1,
        "sampled", "mixed"
    ),
)


def _npu_is_available() -> bool:
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except RuntimeError:
        return False


def _standard_case(nnz: int, query_count: int) -> WorkloadCase:
    return WorkloadCase(
        f"nnz_{nnz}",
        DEFAULT_NUM_ROWS,
        DEFAULT_NUM_COLS,
        nnz,
        query_count,
        query_count,
        DEFAULT_MATCHES_PER_QUERY,
    )


def _pair_from_key(
    keys: torch.Tensor, num_cols: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.div(keys, num_cols, rounding_mode="floor"), keys % num_cols


def _balanced_counts(num_rows: int, nnz: int) -> torch.Tensor:
    quotient, remainder = divmod(nnz, num_rows)
    counts = torch.full((num_rows,), quotient, dtype=torch.int64)
    counts[:remainder] += 1
    return counts


def _patterned_row_counts(case: WorkloadCase) -> torch.Tensor:
    if case.row_pattern == "balanced":
        return _balanced_counts(case.num_rows, case.nnz)
    if case.row_pattern == "sampled":
        hot_rows = max(1, case.num_rows // 10)
        hot_nnz = case.nnz * 4 // 5
        counts = torch.zeros(case.num_rows, dtype=torch.int64)
        counts[:hot_rows] = _balanced_counts(hot_rows, hot_nnz)
        cold_rows = case.num_rows - hot_rows
        if cold_rows:
            counts[hot_rows:] = _balanced_counts(
                cold_rows, case.nnz - hot_nnz
            )
        return counts
    raise ValueError(f"{case.name}: unknown row pattern {case.row_pattern}")


def _patterned_columns(
    case: WorkloadCase, positions: torch.Tensor
) -> torch.Tensor:
    if case.col_pattern == "mixed":
        return (positions * 104_729 + 8_191).remainder(case.num_cols)
    if case.col_pattern == "descending":
        return case.num_cols - 1 - positions.remainder(case.num_cols)
    raise ValueError(f"{case.name}: unknown column pattern {case.col_pattern}")


def _make_patterned_cpu_workload(
    case: WorkloadCase, dtype: torch.dtype
) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
    counts = _patterned_row_counts(case)
    edge_rows = torch.repeat_interleave(
        torch.arange(case.num_rows, dtype=torch.int64), counts
    )
    positions = torch.arange(case.nnz, dtype=torch.int64)
    edge_cols = _patterned_columns(case, positions)

    hit_indices = torch.div(
        torch.arange(case.hit_query_count, dtype=torch.int64) * case.nnz,
        case.hit_query_count,
        rounding_mode="floor",
    )
    query_rows = edge_rows[hit_indices]
    query_cols = edge_cols[hit_indices]

    miss_count = case.query_count - case.hit_query_count
    if miss_count:
        indptr = torch.empty(case.num_rows + 1, dtype=torch.int64)
        indptr[0] = 0
        indptr[1:] = counts.cumsum(0)
        miss_rows = torch.arange(miss_count, dtype=torch.int64).remainder(
            case.num_rows
        )
        miss_positions = indptr[miss_rows + 1]
        miss_cols = _patterned_columns(case, miss_positions)
        query_rows = torch.cat((query_rows, miss_rows))
        query_cols = torch.cat((query_cols, miss_cols))

    graph = dgl.heterograph(
        {("_U", "_E", "_V"): (edge_rows.to(dtype), edge_cols.to(dtype))},
        num_nodes_dict={"_U": case.num_rows, "_V": case.num_cols},
        idtype=dtype,
    ).formats("coo")
    return graph, query_rows.to(dtype), query_cols.to(dtype)


def _make_cpu_workload(
    case: WorkloadCase, dtype: torch.dtype
) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
    if case.row_pattern != "fixed_degree":
        return _make_patterned_cpu_workload(case, dtype)

    coordinate_count = case.num_rows * case.num_cols
    if case.query_count >= coordinate_count:
        raise ValueError(
            f"{case.name}: query_count must leave one unqueried coordinate"
        )
    if not 0 <= case.hit_query_count <= case.query_count:
        raise ValueError(f"{case.name}: invalid hit_query_count")
    if not 1 <= case.matches_per_hit <= DEFAULT_MATCHES_PER_QUERY:
        raise ValueError(
            f"{case.name}: matches_per_hit must be in "
            f"[1, {DEFAULT_MATCHES_PER_QUERY}]"
        )

    edge = torch.arange(case.nnz, dtype=torch.int64)
    matched_nnz = case.expected_outputs
    matched = edge < matched_nnz
    hit_divisor = max(case.hit_query_count, 1)
    matched_keys = edge.remainder(hit_divisor)
    reserved_key = torch.full_like(edge, coordinate_count - 1)
    edge_keys = torch.where(matched, matched_keys, reserved_key)
    edge_rows, edge_cols = _pair_from_key(edge_keys, case.num_cols)

    query_keys = torch.arange(case.query_count, dtype=torch.int64)
    query_rows, query_cols = _pair_from_key(query_keys, case.num_cols)
    graph = dgl.heterograph(
        {("_U", "_E", "_V"): (edge_rows.to(dtype), edge_cols.to(dtype))},
        num_nodes_dict={"_U": case.num_rows, "_V": case.num_cols},
        idtype=dtype,
    ).formats("coo")
    return graph, query_rows.to(dtype), query_cols.to(dtype)


def _invoke(
    graph: dgl.DGLGraph, query_rows: torch.Tensor, query_cols: torch.Tensor
):
    return graph._graph.edge_ids_all(
        graph.get_etype_id(("_U", "_E", "_V")), query_rows, query_cols
    )


def _entries(result: Sequence[torch.Tensor]):
    host_values = [value.detach().cpu().tolist() for value in result]
    return sorted(zip(*host_values))


def _check_results(
    case: WorkloadCase,
    cpu_result: Sequence[torch.Tensor],
    npu_result: Sequence[torch.Tensor],
) -> None:
    cpu_sizes = tuple(value.numel() for value in cpu_result)
    npu_sizes = tuple(value.numel() for value in npu_result)
    expected_sizes = (case.expected_outputs,) * 3
    if cpu_sizes != expected_sizes:
        raise AssertionError(
            f"{case.name}: CPU sizes {cpu_sizes}, expected {expected_sizes}"
        )
    if npu_sizes != expected_sizes:
        raise AssertionError(
            f"{case.name}: NPU sizes {npu_sizes}, expected {expected_sizes}"
        )
    cpu_entries = _entries(cpu_result)
    npu_entries = _entries(npu_result)
    if cpu_entries != npu_entries:
        for index, (cpu_entry, npu_entry) in enumerate(
            zip(cpu_entries, npu_entries)
        ):
            if cpu_entry != npu_entry:
                raise AssertionError(
                    f"{case.name}: first mismatch at sorted index {index}: "
                    f"CPU={cpu_entry}, NPU={npu_entry}"
                )
        raise AssertionError(f"{case.name}: CPU/NPU result multisets differ")


def _benchmark(
    operation: Callable[[], Sequence[torch.Tensor]],
    synchronize: Callable[[], None],
    warmup: int,
    repeat: int,
    batches: int,
) -> Tuple[Timing, Sequence[torch.Tensor]]:
    result = None
    for _ in range(warmup):
        result = operation()
    synchronize()

    samples = []
    for _ in range(batches):
        synchronize()
        start = time.perf_counter()
        for _ in range(repeat):
            result = operation()
        synchronize()
        samples.append((time.perf_counter() - start) * 1_000.0 / repeat)

    if result is None:
        raise AssertionError("benchmark did not execute the operation")
    timing = Timing(
        median_ms=statistics.median(samples),
        minimum_ms=min(samples),
        maximum_ms=max(samples),
        samples_ms=tuple(samples),
    )
    return timing, result


def _run_case(
    case: WorkloadCase,
    dtype_name: str,
    npu_device: torch.device,
    warmup: int,
    repeat: int,
    batches: int,
) -> Comparison:
    cpu_graph, cpu_rows, cpu_cols = _make_cpu_workload(
        case, DTYPES[dtype_name]
    )
    npu_graph = cpu_graph.to(npu_device)
    npu_rows = cpu_rows.to(npu_device)
    npu_cols = cpu_cols.to(npu_device)

    # Correctness setup calls are deliberately outside both timed regions.
    cpu_result = _invoke(cpu_graph, cpu_rows, cpu_cols)
    npu_result = _invoke(npu_graph, npu_rows, npu_cols)
    torch.npu.synchronize(npu_device)
    _check_results(case, cpu_result, npu_result)

    cpu_timing, cpu_result = _benchmark(
        lambda: _invoke(cpu_graph, cpu_rows, cpu_cols),
        lambda: None,
        warmup,
        repeat,
        batches,
    )
    npu_timing, npu_result = _benchmark(
        lambda: _invoke(npu_graph, npu_rows, npu_cols),
        lambda: torch.npu.synchronize(npu_device),
        warmup,
        repeat,
        batches,
    )
    _check_results(case, cpu_result, npu_result)

    comparison = Comparison(
        case=case.name,
        dtype=dtype_name,
        nnz=case.nnz,
        queries=case.query_count,
        outputs=case.expected_outputs,
        cpu_ms=cpu_timing.median_ms,
        npu_ms=npu_timing.median_ms,
        cpu_min_ms=cpu_timing.minimum_ms,
        cpu_max_ms=cpu_timing.maximum_ms,
        npu_min_ms=npu_timing.minimum_ms,
        npu_max_ms=npu_timing.maximum_ms,
        cpu_over_npu=cpu_timing.median_ms / npu_timing.median_ms,
    )
    print(
        f"CASE_RESULT name={case.name} dtype={dtype_name} "
        f"nnz={case.nnz} queries={case.query_count} "
        f"outputs={case.expected_outputs} correctness=PASS "
        f"cpu_ms={comparison.cpu_ms:.6f} "
        f"npu_ms={comparison.npu_ms:.6f} "
        f"cpu_over_npu={comparison.cpu_over_npu:.6f} "
        f"cpu_samples_ms={','.join(f'{x:.6f}' for x in cpu_timing.samples_ms)} "
        f"npu_samples_ms={','.join(f'{x:.6f}' for x in npu_timing.samples_ms)}",
        flush=True,
    )
    return comparison


def _print_summary(results: Sequence[Comparison]) -> None:
    print("\nCPU/NPU summary (cpu_over_npu > 1 means NPU is faster)")
    print(
        f"{'case':<19} {'dtype':<6} {'NNZ':>9} {'Q':>7} "
        f"{'outputs':>9} {'CPU ms':>11} {'NPU ms':>11} {'CPU/NPU':>9}"
    )
    for result in results:
        print(
            f"{result.case:<19} {result.dtype:<6} {result.nnz:>9} "
            f"{result.queries:>7} {result.outputs:>9} "
            f"{result.cpu_ms:>11.6f} {result.npu_ms:>11.6f} "
            f"{result.cpu_over_npu:>9.3f}"
        )


def _write_csv(path: Path, results: Sequence[Comparison]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    print(f"CSV_RESULT path={path.resolve()}", flush=True)


def run_comparison(
    cases: Iterable[WorkloadCase],
    dtype_names: Iterable[str],
    device_id: int,
    warmup: int,
    repeat: int,
    batches: int,
) -> Sequence[Comparison]:
    if not _npu_is_available():
        raise RuntimeError("Ascend NPU is not available")
    npu_device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(npu_device)
    results = []
    for dtype_name in dtype_names:
        for case in cases:
            results.append(
                _run_case(
                    case,
                    dtype_name,
                    npu_device,
                    warmup,
                    repeat,
                    batches,
                )
            )
    return results


def test_coo_get_data_and_indices_cpu_npu():
    """Pytest entry: compare correctness and one timing sample at four scales."""
    import pytest

    if not _npu_is_available():
        pytest.skip("Ascend NPU is not available")
    cases = [_standard_case(nnz, DEFAULT_QUERY_COUNT) for nnz in STANDARD_NNZ]
    results = run_comparison(cases, ("int32",), 0, 1, 1, 1)
    assert len(results) == len(STANDARD_NNZ)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("standard", "large", "all"),
        default="standard",
        help="standard: four NNZ scales; large: former GPU-scale cases",
    )
    parser.add_argument(
        "--nnz",
        type=_positive,
        action="append",
        help="Standard-suite NNZ; repeat to select multiple. Defaults to four scales.",
    )
    parser.add_argument("--queries", type=_positive, default=DEFAULT_QUERY_COUNT)
    parser.add_argument(
        "--dtype", choices=("int32", "int64", "both"), default="int32"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=_positive, default=10)
    parser.add_argument("--batches", type=_positive, default=5)
    parser.add_argument("--csv", type=Path, help="Optional summary CSV path")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be >= 0")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.queries >= DEFAULT_NUM_ROWS * DEFAULT_NUM_COLS:
        parser.error("--queries is too large for the standard graph shape")
    return args


def main() -> None:
    args = _parse_args()
    cases = []
    if args.suite in ("standard", "all"):
        nnz_values = tuple(dict.fromkeys(args.nnz or STANDARD_NNZ))
        cases.extend(_standard_case(nnz, args.queries) for nnz in nnz_values)
    if args.suite in ("large", "all"):
        cases.extend(LARGE_CASES)
    dtype_names = tuple(DTYPES) if args.dtype == "both" else (args.dtype,)
    results = run_comparison(
        cases,
        dtype_names,
        args.device,
        args.warmup,
        args.repeat,
        args.batches,
    )
    _print_summary(results)
    if args.csv:
        _write_csv(args.csv, results)


if __name__ == "__main__":
    main()
