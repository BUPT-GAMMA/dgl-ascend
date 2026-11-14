"""msprof benchmark for COOGetRowDataAndIndices on Ascend NPU.

The benchmark reaches the native operator through DGL's scalar in-edges CAPI
on a graph whose only allowed sparse format is COO.  The scalar C++ ``InEdges``
overload transposes the COO view and calls ``COOGetRowDataAndIndices``; the two
kernels of interest in the msprof report are therefore:

* ``coo_get_row_count_i32``
* ``coo_get_row_scatter_i32``

The default workload uses int32 IDs and makes every eighth edge an incoming
edge of node 0, so every case has a deterministic 12.5% match rate and runs
both the count and scatter phases.  Use ``--dtype int64`` to profile the i64
kernel variants instead.

Collect the four requested NNZ cases in separate msprof directories:

    PYTHONPATH=python python tests/ascend/coo_get_row_data_and_indices_perf.py \
        --run-msprof --msprof-output ./msprof_coo_get_row \
        --warmup 5 --repeat 20

Profile one case directly:

    msprof --output=./msprof_coo_get_row/nnz_10000_int32 \
        --ascendcl=on --runtime-api=on --task-time=l2 --ai-core=on \
        --aic-metrics=PipeUtilization \
        python tests/ascend/coo_get_row_data_and_indices_perf.py \
        --performance --nnz 10000 --warmup 5 --repeat 20

The printed wall time is the scalar DGL CAPI call's end-to-end latency.  After
each collection the launcher also reads the latest msprof ``op_statistic.csv``
and prints the count/scatter device-kernel averages.  The graph construction
and warmup are outside the printed timed interval, although msprof still
records them in the process trace.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
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
from dgl import backend as F  # noqa: E402
from dgl import heterograph_index  # noqa: E402


NNZ_CASES: Tuple[int, ...] = (1_000, 10_000, 50_000, 100_000)
TARGET_ROW = 0
MATCH_STRIDE = 8
NUM_NODES = 4_096
DTYPES = {"int32": torch.int32, "int64": torch.int64}


def _npu_is_available() -> bool:
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except RuntimeError:
        return False


def _make_coo_graph(
    nnz: int, dtype: torch.dtype, device: torch.device
) -> dgl.DGLGraph:
    """Build a deterministic, unsorted COO graph with a fixed match rate."""
    edge = torch.arange(nnz, dtype=torch.int64)
    src = ((edge * 17 + 3) % NUM_NODES).to(dtype)

    # Keep all regular destinations away from TARGET_ROW, then insert an exact
    # target match every MATCH_STRIDE edges.  Matches are distributed across
    # all input chunks instead of being concentrated in one block.
    dst = ((edge * 29 + 7) % (NUM_NODES - 1) + 1).to(dtype)
    dst[::MATCH_STRIDE] = TARGET_ROW

    graph = dgl.graph((src, dst), num_nodes=NUM_NODES, idtype=dtype)
    return graph.formats("coo").to(device)


def _invoke(graph: dgl.DGLGraph):
    """Call the scalar UnitGraph::InEdges overload that owns this operator.

    The public Python ``graph.in_edges`` API converts even a scalar vertex to a
    one-element tensor and therefore selects the multi-row ``COOSliceRows``
    overload.  Calling the scalar CAPI is intentional here: it isolates the
    production C++ path that invokes ``COOGetRowDataAndIndices``.
    """
    edge_array = heterograph_index._CAPI_DGLHeteroInEdges_1(
        graph._graph, graph.get_etype_id(None), TARGET_ROW
    )
    return tuple(F.from_dgl_nd(edge_array(i)) for i in range(3))


def _check_result_shape(result, nnz: int) -> int:
    src, dst, eid = result
    expected_matches = (nnz + MATCH_STRIDE - 1) // MATCH_STRIDE
    actual_shapes = (src.numel(), dst.numel(), eid.numel())
    if actual_shapes != (expected_matches,) * 3:
        raise AssertionError(
            f"unexpected result sizes for nnz={nnz}: got {actual_shapes}, "
            f"expected {(expected_matches,) * 3}"
        )
    return expected_matches


def _run_case(
    nnz: int,
    dtype_name: str,
    device_id: int,
    warmup: int,
    repeat: int,
) -> float:
    if not _npu_is_available():
        raise RuntimeError("Ascend NPU is not available")

    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    graph = _make_coo_graph(nnz, DTYPES[dtype_name], device)

    result = None
    for _ in range(warmup):
        result = _invoke(graph)
    torch.npu.synchronize(device)

    # Validate metadata without copying output tensors back to the host.
    if result is None:
        result = _invoke(graph)
        torch.npu.synchronize(device)
    matches = _check_result_shape(result, nnz)

    print(
        f"PERF_CONFIG nnz={nnz} matches={matches} "
        f"match_rate={matches / nnz:.6f} dtype={dtype_name} "
        f"device={device_id} warmup={warmup} repeat={repeat}",
        flush=True,
    )
    print(f"PERF_START nnz={nnz}", flush=True)
    start = time.perf_counter()
    for _ in range(repeat):
        result = _invoke(graph)
    torch.npu.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1_000.0

    # Keep the final result live through synchronization and check that every
    # profiled invocation followed the non-empty scatter path.
    _check_result_shape(result, nnz)
    average_ms = elapsed_ms / repeat
    throughput_mnnz_s = nnz * repeat / (elapsed_ms * 1_000.0)
    print(
        f"PERF_END nnz={nnz} total_ms={elapsed_ms:.6f} "
        f"average_ms={average_ms:.6f} "
        f"throughput_mnnz_s={throughput_mnnz_s:.6f}",
        flush=True,
    )
    return average_ms


def _run_msprof(
    output_dir: Path,
    nnz_cases: Sequence[int],
    dtype_name: str,
    device_id: int,
    warmup: int,
    repeat: int,
) -> None:
    msprof = shutil.which("msprof")
    if msprof is None:
        raise RuntimeError(
            "msprof was not found in PATH; source the Ascend toolkit "
            "environment before running this script"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    for nnz in nnz_cases:
        case_output = output_dir / f"nnz_{nnz}_{dtype_name}"
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
        ]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        _print_msprof_summary(case_output, nnz, dtype_name)


def _print_msprof_summary(
    case_output: Path, nnz: int, dtype_name: str
) -> None:
    """Print count/scatter averages from the newest exported msprof CSV."""
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
    suffix = "i32" if dtype_name == "int32" else "i64"
    kernel_names = (
        f"coo_get_row_count_{suffix}",
        f"coo_get_row_scatter_{suffix}",
    )
    with report.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = {row["OP Type"]: row for row in csv.DictReader(csv_file)}

    averages = []
    for kernel_name in kernel_names:
        row = rows.get(kernel_name)
        if row is None:
            raise RuntimeError(
                f"{kernel_name} is missing from msprof report {report}"
            )
        average_us = float(row["Avg Time(us)"])
        averages.append(average_us)
        print(
            f"MSPROF_KERNEL nnz={nnz} kernel={kernel_name} "
            f"count={row['Count']} avg_us={average_us:.6f} "
            f"min_us={float(row['Min Time(us)']):.6f} "
            f"max_us={float(row['Max Time(us)']):.6f}",
            flush=True,
        )
    kernel_total_us = sum(averages)
    print(
        f"MSPROF_RESULT nnz={nnz} count_plus_scatter_us="
        f"{kernel_total_us:.6f} "
        f"kernel_throughput_mnnz_s={nnz / kernel_total_us:.6f} "
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
        "--msprof-output",
        type=Path,
        default=Path("./msprof_coo_get_row"),
    )
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be >= 0")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0")
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
        )
    else:
        for selected_case in selected_nnz:
            _run_case(
                selected_case,
                cli_args.dtype,
                cli_args.device,
                cli_args.warmup,
                cli_args.repeat,
            )
