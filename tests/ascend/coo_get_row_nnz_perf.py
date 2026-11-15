"""Vector-query msprof benchmark for COOGetRowNNZ on Ascend NPU.

Performance measurements use the array ``InDegrees`` overload on a graph
whose only allowed sparse format is COO.  Scalar ``InDegree`` is called once
outside the timed region for accuracy only.  The kernel of interest in the
msprof report is:

* ``coo_get_row_nnz_int32``

The default workload uses int32 IDs, 128 query rows, and makes every eighth
edge an incoming edge of node 0.  Use ``--query-count`` to change the vector
length and ``--dtype int64`` to profile ``coo_get_row_nnz_int64`` instead.

Collect the four requested NNZ cases in separate msprof directories:

    PYTHONPATH=python python tests/ascend/coo_get_row_nnz_perf.py \
        --run-msprof --msprof-output ./msprof_coo_get_row_nnz \
        --warmup 5 --repeat 20

Profile one case directly:

    msprof --output=./msprof_coo_get_row_nnz/nnz_10000_int32 \
        --ascendcl=on --runtime-api=on --task-time=l2 --ai-core=on \
        --aic-metrics=PipeUtilization \
        python tests/ascend/coo_get_row_nnz_perf.py \
        --performance --nnz 10000 --warmup 5 --repeat 20

The printed CPU and NPU wall times are vector DGL CAPI end-to-end latencies.
Vector results are checked bit-for-bit, while the scalar path is checked once
without being timed.  During ``--run-msprof``, scalar accuracy runs before the
profiler starts so the same-named scalar kernel cannot contaminate the vector
kernel average.  Graph construction, query construction, result validation,
and warmup are outside the printed timed intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence, Tuple

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

# DGL-Ascend requires the PyTorch and torch_npu shared libraries to be loaded
# before libdgl.so is opened.
import dgl  # noqa: E402
from dgl import heterograph_index  # noqa: E402


NNZ_CASES: Tuple[int, ...] = (1_000, 10_000, 50_000, 100_000)
TARGET_ROW = 0
MATCH_STRIDE = 8
NUM_NODES = 4_096
DEFAULT_QUERY_COUNT = 128
DTYPES = {"int32": torch.int32, "int64": torch.int64}


def _npu_is_available() -> bool:
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except RuntimeError:
        return False


def _make_coo_graphs(
    nnz: int, dtype: torch.dtype, device: torch.device
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
    """Build CPU and NPU COO graphs backed by the same deterministic data."""
    edge = torch.arange(nnz, dtype=torch.int64)
    src = ((edge * 17 + 3) % NUM_NODES).to(dtype)

    # Keep regular destinations away from TARGET_ROW and insert one exact
    # target match every MATCH_STRIDE edges.  Matches are distributed across
    # all input tiles instead of being concentrated in one tile.
    dst = ((edge * 29 + 7) % (NUM_NODES - 1) + 1).to(dtype)
    dst[::MATCH_STRIDE] = TARGET_ROW

    cpu_graph = dgl.graph(
        (src, dst), num_nodes=NUM_NODES, idtype=dtype
    ).formats("coo")
    return cpu_graph, cpu_graph.to(device)


def _invoke_scalar(graph: dgl.DGLGraph) -> int:
    """Call the scalar overload for accuracy validation only.

    Calling the scalar CAPI directly selects
    ``COOGetRowNNZ(COOMatrix, int64_t)``.
    """
    return int(
        heterograph_index._CAPI_DGLHeteroInDegree(
            graph._graph, graph.get_etype_id(None), TARGET_ROW
        )
    )


def _invoke_vector(
    graph: dgl.DGLGraph, queries: torch.Tensor
) -> torch.Tensor:
    """Call the array overload without the public API's defensive retry."""
    return graph._graph.in_degrees(graph.get_etype_id(None), queries)


def _make_queries(
    query_count: int, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    cpu_queries = (
        torch.arange(query_count, dtype=torch.int64) % NUM_NODES
    ).to(dtype)
    return cpu_queries, cpu_queries.to(device)


def _check_scalar_results(
    cpu_result: int, npu_result: int, nnz: int
) -> int:
    expected_matches = (nnz + MATCH_STRIDE - 1) // MATCH_STRIDE
    if cpu_result != expected_matches or npu_result != expected_matches:
        raise AssertionError(
            f"row nnz mismatch for nnz={nnz}: CPU={cpu_result}, "
            f"NPU={npu_result}, expected={expected_matches}"
        )
    return expected_matches


def _check_vector_results(
    cpu_result: torch.Tensor,
    npu_result: torch.Tensor,
    nnz: int,
    query_count: int,
) -> int:
    npu_cpu = npu_result.cpu()
    if not torch.equal(cpu_result, npu_cpu):
        mismatch = torch.nonzero(cpu_result != npu_cpu).flatten()
        first = int(mismatch[0]) if mismatch.numel() else -1
        raise AssertionError(
            f"vector row nnz mismatch for nnz={nnz}, "
            f"query_count={query_count}, first_mismatch={first}"
        )

    expected_target_matches = (nnz + MATCH_STRIDE - 1) // MATCH_STRIDE
    if int(cpu_result[0]) != expected_target_matches:
        raise AssertionError(
            f"target row mismatch for nnz={nnz}: "
            f"result={int(cpu_result[0])}, "
            f"expected={expected_target_matches}"
        )
    return int(cpu_result.to(torch.int64).sum())


def _check_scalar_accuracy(
    cpu_graph: dgl.DGLGraph,
    npu_graph: dgl.DGLGraph,
    nnz: int,
    dtype_name: str,
) -> Tuple[int, int]:
    cpu_result = _invoke_scalar(cpu_graph)
    npu_result = _invoke_scalar(npu_graph)
    matches = _check_scalar_results(cpu_result, npu_result, nnz)
    print(
        f"SCALAR_ACCURACY_RESULT nnz={nnz} dtype={dtype_name} "
        f"status=PASS exact_match=true cpu_result={cpu_result} "
        f"npu_result={npu_result} expected={matches}",
        flush=True,
    )
    return cpu_result, npu_result


def _run_scalar_accuracy_case(
    nnz: int, dtype_name: str, device_id: int
) -> None:
    if not _npu_is_available():
        raise RuntimeError("Ascend NPU is not available")
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    cpu_graph, npu_graph = _make_coo_graphs(
        nnz, DTYPES[dtype_name], device
    )
    _check_scalar_accuracy(cpu_graph, npu_graph, nnz, dtype_name)


def _run_case(
    nnz: int,
    dtype_name: str,
    device_id: int,
    warmup: int,
    repeat: int,
    query_count: int,
    check_scalar_accuracy: bool,
) -> dict:
    if not _npu_is_available():
        raise RuntimeError("Ascend NPU is not available")

    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    cpu_graph, npu_graph = _make_coo_graphs(
        nnz, DTYPES[dtype_name], device
    )
    cpu_queries, npu_queries = _make_queries(
        query_count, DTYPES[dtype_name], device
    )

    if check_scalar_accuracy:
        _check_scalar_accuracy(cpu_graph, npu_graph, nnz, dtype_name)

    # Validate the vector overload once outside the timed region.
    cpu_result = _invoke_vector(cpu_graph, cpu_queries)
    npu_result = _invoke_vector(npu_graph, npu_queries)
    torch.npu.synchronize(device)
    vector_checksum = _check_vector_results(
        cpu_result, npu_result, nnz, query_count
    )
    print(
        f"VECTOR_ACCURACY_RESULT nnz={nnz} dtype={dtype_name} "
        f"query_count={query_count} status=PASS exact_match=true "
        f"result_checksum={vector_checksum}",
        flush=True,
    )

    for _ in range(warmup):
        cpu_result = _invoke_vector(cpu_graph, cpu_queries)
        npu_result = _invoke_vector(npu_graph, npu_queries)
    torch.npu.synchronize(device)

    print(
        f"PERF_CONFIG nnz={nnz} query_count={query_count} "
        f"work_items={nnz * query_count} dtype={dtype_name} "
        f"device={device_id} warmup={warmup} repeat={repeat}",
        flush=True,
    )

    print(f"PERF_START nnz={nnz} device=npu", flush=True)
    start = time.perf_counter()
    for _ in range(repeat):
        npu_result = _invoke_vector(npu_graph, npu_queries)
    torch.npu.synchronize(device)
    npu_elapsed_ms = (time.perf_counter() - start) * 1_000.0

    npu_average_ms = npu_elapsed_ms / repeat
    npu_throughput_mcomparisons_s = (
        nnz * query_count * repeat / (npu_elapsed_ms * 1_000.0)
    )
    print(
        f"PERF_END nnz={nnz} device=npu total_ms={npu_elapsed_ms:.6f} "
        f"average_ms={npu_average_ms:.6f} "
        f"throughput_mcomparisons_s="
        f"{npu_throughput_mcomparisons_s:.6f}",
        flush=True,
    )

    print(f"PERF_START nnz={nnz} device=cpu", flush=True)
    start = time.perf_counter()
    for _ in range(repeat):
        cpu_result = _invoke_vector(cpu_graph, cpu_queries)
    cpu_elapsed_ms = (time.perf_counter() - start) * 1_000.0

    cpu_average_ms = cpu_elapsed_ms / repeat
    cpu_throughput_mcomparisons_s = (
        nnz * query_count * repeat / (cpu_elapsed_ms * 1_000.0)
    )
    print(
        f"PERF_END nnz={nnz} device=cpu total_ms={cpu_elapsed_ms:.6f} "
        f"average_ms={cpu_average_ms:.6f} "
        f"throughput_mcomparisons_s="
        f"{cpu_throughput_mcomparisons_s:.6f}",
        flush=True,
    )

    vector_checksum = _check_vector_results(
        cpu_result, npu_result, nnz, query_count
    )
    npu_speedup = cpu_average_ms / npu_average_ms
    print(
        f"PERF_COMPARISON nnz={nnz} dtype={dtype_name} "
        f"query_count={query_count} vector_accuracy=PASS "
        f"cpu_average_ms={cpu_average_ms:.6f} "
        f"npu_average_ms={npu_average_ms:.6f} "
        f"npu_speedup_vs_cpu={npu_speedup:.6f} "
        f"cpu_throughput_mcomparisons_s="
        f"{cpu_throughput_mcomparisons_s:.6f} "
        f"npu_throughput_mcomparisons_s="
        f"{npu_throughput_mcomparisons_s:.6f}",
        flush=True,
    )
    return {
        "nnz": nnz,
        "dtype": dtype_name,
        "query_count": query_count,
        "vector_checksum": vector_checksum,
        "cpu_average_ms": cpu_average_ms,
        "npu_average_ms": npu_average_ms,
        "npu_speedup": npu_speedup,
        "cpu_throughput_mcomparisons_s": (
            cpu_throughput_mcomparisons_s
        ),
        "npu_throughput_mcomparisons_s": (
            npu_throughput_mcomparisons_s
        ),
    }


def _print_comparison_summary(results: Sequence[dict]) -> None:
    print("CPU_NPU_COMPARISON_SUMMARY", flush=True)
    for result in results:
        print(
            f"CPU_NPU_COMPARISON nnz={result['nnz']} "
            f"dtype={result['dtype']} query_count={result['query_count']} "
            f"vector_accuracy=PASS "
            f"result_checksum={result['vector_checksum']} "
            f"cpu_average_ms={result['cpu_average_ms']:.6f} "
            f"npu_average_ms={result['npu_average_ms']:.6f} "
            f"npu_speedup_vs_cpu={result['npu_speedup']:.6f}",
            flush=True,
        )


def _run_msprof(
    output_dir: Path,
    nnz_cases: Sequence[int],
    dtype_name: str,
    device_id: int,
    warmup: int,
    repeat: int,
    query_count: int,
) -> None:
    msprof = shutil.which("msprof")
    if msprof is None:
        raise RuntimeError(
            "msprof was not found in PATH; source the Ascend toolkit "
            "environment before running this script"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    comparison_results = []
    with tempfile.TemporaryDirectory(
        prefix="coo_get_row_nnz_results_"
    ) as result_dir:
        for nnz in nnz_cases:
            # Keep the scalar kernel out of the profiler's same-name operator
            # aggregate.  Scalar is accuracy-only; vector owns all timings.
            _run_scalar_accuracy_case(nnz, dtype_name, device_id)

            case_output = output_dir / f"nnz_{nnz}_{dtype_name}"
            result_file = Path(result_dir) / f"nnz_{nnz}.json"
            command = [
                msprof,
                f"--output={case_output}",
                "--ascendcl=on",
                "--runtime-api=on",
                "--task-time=l2",
                "--ai-core=on",
                "--aic-metrics=PipeUtilization",
                sys.executable,
                str(script),
                "--performance",
                "--nnz",
                str(nnz),
                "--dtype",
                dtype_name,
                "--device",
                str(device_id),
                "--warmup",
                str(warmup),
                "--repeat",
                str(repeat),
                "--query-count",
                str(query_count),
                "--skip-scalar-accuracy",
                "--result-file",
                str(result_file),
            ]
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, check=True)
            comparison_results.extend(
                json.loads(result_file.read_text(encoding="utf-8"))
            )
            _print_msprof_summary(
                case_output, nnz, dtype_name, query_count
            )
    _print_comparison_summary(comparison_results)


def _print_msprof_summary(
    case_output: Path,
    nnz: int,
    dtype_name: str,
    query_count: int,
) -> None:
    """Print the COOGetRowNNZ average from the newest exported CSV."""
    reports = list(
        case_output.glob(
            "PROF_*/mindstudio_profiler_output/op_statistic_*.csv"
        )
    )
    if not reports:
        print(
            f"MSPROF_WARNING nnz={nnz} op_statistic.csv was not found",
            flush=True,
        )
        return

    report = max(reports, key=lambda path: path.stat().st_mtime_ns)
    kernel_name = f"coo_get_row_nnz_{dtype_name}"
    with report.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = {row["OP Type"]: row for row in csv.DictReader(csv_file)}

    row = rows.get(kernel_name)
    if row is None:
        observed = ", ".join(sorted(rows)) or "<none>"
        raise RuntimeError(
            f"{kernel_name} is missing from msprof report {report}; "
            f"observed OP Type values: {observed}. Make sure libdgl.so was "
            "rebuilt after the COOGetRowNNZ Ascend implementation changed."
        )

    average_us = float(row["Avg Time(us)"])
    print(
        f"MSPROF_KERNEL nnz={nnz} kernel={kernel_name} "
        f"count={row['Count']} avg_us={average_us:.6f} "
        f"min_us={float(row['Min Time(us)']):.6f} "
        f"max_us={float(row['Max Time(us)']):.6f}",
        flush=True,
    )
    print(
        f"MSPROF_RESULT nnz={nnz} kernel_us={average_us:.6f} "
        f"query_count={query_count} "
        f"kernel_throughput_mcomparisons_s="
        f"{nnz * query_count / average_us:.6f} "
        f"report={report}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--performance",
        action="store_true",
        help="Run workloads directly, normally as an msprof child process.",
    )
    mode.add_argument(
        "--run-msprof",
        action="store_true",
        help="Launch one msprof collection process per NNZ case.",
    )
    parser.add_argument(
        "--nnz",
        type=int,
        action="append",
        choices=NNZ_CASES,
        help="NNZ case to run; repeat the option to select several. "
        "Defaults to all four cases.",
    )
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="int32")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--query-count",
        type=int,
        default=DEFAULT_QUERY_COUNT,
        help="Number of rows in each timed vector query (default: 128).",
    )
    parser.add_argument(
        "--msprof-output",
        type=Path,
        default=Path("./msprof_coo_get_row_nnz"),
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-scalar-accuracy",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be >= 0")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0")
    if args.query_count <= 0:
        parser.error("--query-count must be > 0")
    return args


if __name__ == "__main__":
    cli_args = _parse_args()
    selected_nnz = tuple(dict.fromkeys(cli_args.nnz or NNZ_CASES))
    if cli_args.run_msprof:
        _run_msprof(
            cli_args.msprof_output,
            selected_nnz,
            cli_args.dtype,
            cli_args.device,
            cli_args.warmup,
            cli_args.repeat,
            cli_args.query_count,
        )
    else:
        comparison_results = []
        for selected_case in selected_nnz:
            comparison_results.append(
                _run_case(
                    selected_case,
                    cli_args.dtype,
                    cli_args.device,
                    cli_args.warmup,
                    cli_args.repeat,
                    cli_args.query_count,
                    not cli_args.skip_scalar_accuracy,
                )
            )
        _print_comparison_summary(comparison_results)
        if cli_args.result_file is not None:
            cli_args.result_file.write_text(
                json.dumps(comparison_results), encoding="utf-8"
            )
