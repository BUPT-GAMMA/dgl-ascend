"""Validate and benchmark native CSR SDDMM on the four requested layouts.

    PYTHONPATH=python python3 tests/ascend/bench_sddmm_csr_npu.py --profile

All inputs and CSR graphs are prepared before timing. ``api_*`` measures the
synchronized internal ``dgl._sparse_ops._gsddmm`` entry point, including output
allocation, dispatch and tiling copies. This directly exercises the native
C++ entry independent of backend autograd/autocast rewrites.
``kernel_*`` comes from a separate CANN profiler run, never from host timing or
NPU events around synchronous host work. Every output element is checked
against an independent, chunked CPU reference before performance sampling.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

# Register Ascend support before loading libdgl.so.
import dgl
from dgl._sparse_ops import _gsddmm
from dgl.ops.sddmm import reshape_lhs_rhs


OPS = ("add", "sub", "mul", "div", "copy_lhs", "copy_rhs", "dot")
TARGETS = ("u", "e", "v")
ETYPE = ("source", "edge", "destination")


@dataclass(frozen=True)
class Case:
    name: str
    targets: tuple[str, str]
    graph: tuple[int, int, int]
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]


PERFORMANCE_CASES = (
    Case("ev", ("e", "v"), (4096, 2048, 65536), (64,), (64,)),
    Case("vu", ("v", "u"), (10000, 5000, 200000), (128,), (128,)),
    Case("ue", ("u", "e"), (10000, 10000, 500000), (64,), (64,)),
    Case("uv", ("u", "v"), (50000, 5000, 500000), (8, 16), (8, 16)),
)


@dataclass
class PreparedCase:
    case: Case
    graph: dgl.DGLGraph
    lhs_cpu: torch.Tensor
    rhs_cpu: torch.Tensor
    lhs_npu: torch.Tensor
    rhs_npu: torch.Tensor
    rows: dict[str, torch.Tensor]
    edge_ids: torch.Tensor


def npu_available():
    return torch_npu is not None and torch.npu.is_available()


def prepare_case(case, device, idtype=torch.int32, dtype=torch.float32,
                 mapped=False, seed=2026, counts=None):
    """Construct an exact-size CSR graph, with reproducible random edge order."""
    source_nodes, destination_nodes, edges = case.graph
    generator = torch.Generator().manual_seed(seed)
    if counts is None:
        sampled = torch.randint(source_nodes, (edges,), generator=generator)
        counts = torch.bincount(sampled, minlength=source_nodes)
    assert len(counts) == source_nodes and int(counts.sum()) == edges
    rows = torch.repeat_interleave(torch.arange(source_nodes), counts)
    indptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    indices = torch.randint(destination_nodes, (edges,), generator=generator)
    edge_ids = torch.randperm(edges, generator=generator) if mapped else torch.arange(edges)
    data = edge_ids.to(idtype) if mapped else torch.empty(0, dtype=idtype)
    graph = dgl.heterograph(
        {ETYPE: ("csr", (indptr.to(idtype), indices.to(idtype), data))},
        num_nodes_dict={"source": source_nodes, "destination": destination_nodes},
        idtype=idtype,
    ).formats("csr").to(device)
    assert graph.formats()["created"] == ["csr"]
    actual_indptr, actual_indices, actual_data = graph.adj_tensors("csr")
    assert actual_indptr.dtype == idtype and actual_indices.dtype == idtype
    assert actual_data.numel() == (edges if mapped else 0)
    sizes = {"u": source_nodes, "e": edges, "v": destination_nodes}
    lhs = torch.randn((sizes[case.targets[0]],) + case.lhs_shape,
                      generator=generator, dtype=dtype)
    # Avoid unstable near-zero divisors; special values have a separate test.
    rhs = torch.rand((sizes[case.targets[1]],) + case.rhs_shape,
                     generator=generator, dtype=dtype) + 0.5
    return PreparedCase(case, graph, lhs, rhs, lhs.to(device), rhs.to(device),
                        {"u": rows, "e": edge_ids, "v": indices}, edge_ids)


def run_operator(prepared, op, public=False):
    lhs = None if op == "copy_rhs" else prepared.lhs_npu
    rhs = None if op == "copy_lhs" else prepared.rhs_npu
    if public:
        return dgl.ops.gsddmm(prepared.graph, op, lhs, rhs,
                             lhs_target=prepared.case.targets[0],
                             rhs_target=prepared.case.targets[1])
    if lhs is not None and rhs is not None:
        lhs, rhs = reshape_lhs_rhs(lhs, rhs)
    return _gsddmm(prepared.graph._graph, op, lhs, rhs, *prepared.case.targets)


def output_features(case, op):
    if op == "copy_lhs":
        return case.lhs_shape
    if op == "copy_rhs":
        return case.rhs_shape
    features = tuple(torch.broadcast_shapes(case.lhs_shape, case.rhs_shape))
    return features[:-1] + (1,) if op == "dot" else features


def check_accuracy(prepared, op, actual=None):
    """Check all elements in edge-ID order without materializing large gathers."""
    case = prepared.case
    actual = (run_operator(prepared, op) if actual is None else actual).cpu()
    expected_shape = (case.graph[2],) + output_features(case, op)
    assert tuple(actual.shape) == expected_shape, (actual.shape, expected_shape)
    assert actual.dtype == prepared.lhs_cpu.dtype
    assert prepared.graph.formats()["created"] == ["csr"]
    full_shape = (case.lhs_shape if op == "copy_lhs" else case.rhs_shape
                  if op == "copy_rhs" else
                  tuple(torch.broadcast_shapes(case.lhs_shape, case.rhs_shape)))
    chunk_edges = max(1, (1 << 20) // math.prod(full_shape))
    is_double = actual.dtype == torch.float64
    rtol = 1e-12 if is_double else 2e-5
    atol = 1e-12 if is_double else 2e-5
    if op == "dot":
        atol *= max(1.0, math.sqrt(case.lhs_shape[-1]))
    if op in ("copy_lhs", "copy_rhs"):
        rtol = atol = 0
    max_error, elements = 0.0, 0
    for begin in range(0, case.graph[2], chunk_edges):
        end = min(case.graph[2], begin + chunk_edges)
        selected = []
        for slot, (values, target, shape) in enumerate(zip(
                (prepared.lhs_cpu, prepared.rhs_cpu), case.targets,
                (case.lhs_shape, case.rhs_shape))):
            if (op == "copy_lhs" and slot == 1) or (op == "copy_rhs" and slot == 0):
                selected.append(None)
                continue
            value = values[prepared.rows[target][begin:end]]
            selected.append(value.reshape((end - begin,) +
                            (1,) * (len(full_shape) - len(shape)) + shape))
        lhs, rhs = selected
        if op == "copy_lhs":
            expected = lhs
        elif op == "copy_rhs":
            expected = rhs
        elif op == "dot":
            expected = (lhs * rhs).sum(dim=-1, keepdim=True)
        else:
            expected = {"add": torch.add, "sub": torch.sub,
                        "mul": torch.mul, "div": torch.div}[op](lhs, rhs)
        result = actual[prepared.edge_ids[begin:end]]
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
        max_error = max(max_error, float((result - expected).abs().max()))
        elements += expected.numel()
    assert elements == actual.numel()
    return {"status": "PASS", "max_abs_error": max_error,
            "compared_elements": elements, "rtol": rtol, "atol": atol}


def latency_statistics(samples):
    ordered = sorted(samples)
    return {"median_us": statistics.median(samples),
            "mean_us": statistics.mean(samples),
            "p90_us": ordered[math.ceil(len(samples) * 0.9) - 1],
            "min_us": ordered[0], "max_us": ordered[-1]}


def profile_kernels(output_dir, device, configurations, warmup, repeat, seed):
    # Reuse the existing add benchmark's CANN Level1 profiler configuration.
    from sddmm_csr_add_validation import make_profiler

    profile_dir = output_dir / "profile"
    previous = None
    prepared = None
    with make_profiler(profile_dir) as profiler:
        for case_index, mapped, op in configurations:
            if (case_index, mapped) != previous:
                prepared = prepare_case(PERFORMANCE_CASES[case_index], device,
                                        mapped=mapped, seed=seed + case_index)
                previous = case_index, mapped
            for _ in range(warmup + repeat):
                run_operator(prepared, op)
        torch.npu.synchronize(device)
        profiler.step()
    records = []
    for path in profile_dir.glob("**/ASCEND_PROFILER_OUTPUT/kernel_details.csv"):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                name = row["Name"].strip()
                if name.startswith("sddmm_csr_"):
                    records.append((float(row["Start Time(us)"]), name,
                                    float(row["Duration(us)"])))
    records.sort(key=lambda row: row[0])
    expected_count = len(configurations) * (warmup + repeat)
    assert len(records) == expected_count, (
        f"Expected {expected_count} native CSR launches; found {len(records)}. "
        f"Inspect {profile_dir} for a fallback or missing kernel.")
    results = []
    for index in range(len(configurations)):
        start = index * (warmup + repeat)
        samples = records[start + warmup:start + warmup + repeat]
        symbols = sorted({row[1] for row in samples})
        assert len(symbols) == 1, symbols
        results.append((symbols[0], [row[2] for row in samples]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS))
    parser.add_argument("--cases", nargs="+", choices=range(1, 5), type=int,
                        default=[1, 2, 3, 4])
    parser.add_argument("--edge-map", choices=("both", "none", "random"), default="both")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile", action="store_true", help="Collect actual CANN kernel durations")
    parser.add_argument("--output-dir", type=Path, default=Path("build/sddmm_csr_validation"))
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1 or args.threads < 1:
        parser.error("warmup must be nonnegative; repeat and threads must be positive")
    if not npu_available():
        raise RuntimeError("Ascend NPU is required")
    torch.set_num_threads(args.threads)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    maps = (False, True) if args.edge_map == "both" else (args.edge_map == "random",)
    configurations = list(itertools.product((index - 1 for index in args.cases), maps, args.ops))
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True)
    results, samples = [], []
    summary = {
        "status": "running", "started_utc": run_name,
        "seed": args.seed, "warmup": args.warmup, "repeat": args.repeat,
        "api_entry": "dgl._sparse_ops._gsddmm",
        "api_timing": "synchronized wall time; includes output allocation, dispatch, tiling copies",
        "kernel_timing": "separate CANN Level1 kernel_details.csv run" if args.profile else None,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torch_npu": torch_npu.__version__, "dgl": dgl.__version__,
                        "device": str(device), "device_name": torch.npu.get_device_name(args.device)},
        "results": results,
    }
    print(f"Results: {output_dir.resolve()}", flush=True)
    try:
        with torch.inference_mode():
            previous = None
            for case_index, mapped, op in configurations:
                case = PERFORMANCE_CASES[case_index]
                if (case_index, mapped) != previous:
                    prepared = prepare_case(case, device, mapped=mapped, seed=args.seed + case_index)
                    previous = case_index, mapped
                metrics = check_accuracy(prepared, op)
                for _ in range(args.warmup):
                    run_operator(prepared, op)
                torch.npu.synchronize(device)
                api_samples = []
                for _ in range(args.repeat):
                    start = time.perf_counter_ns()
                    output = run_operator(prepared, op)
                    torch.npu.synchronize(device)
                    api_samples.append((time.perf_counter_ns() - start) / 1000)
                    del output
                row = {"case": case.name, "op": op, "M": case.graph[0], "N": case.graph[1],
                       "E": case.graph[2], "lhs_target": case.targets[0], "rhs_target": case.targets[1],
                       "lhs_shape": str(tuple(prepared.lhs_npu.shape)),
                       "rhs_shape": str(tuple(prepared.rhs_npu.shape)),
                       "out_shape": str((case.graph[2],) + output_features(case, op)),
                       "idtype": "int32", "dtype": "float32", "mapped": mapped, **metrics}
                row.update({f"api_{key}": value for key, value in latency_statistics(api_samples).items()})
                results.append(row)
                samples.extend({"case": case.name, "op": op, "mapped": mapped,
                                "metric": "api", "sample": i, "duration_us": value}
                               for i, value in enumerate(api_samples))
                print(f"PASS {case.name} {op:8} map={mapped} API={row['api_median_us']:.3f} us", flush=True)
            if args.profile:
                profiles = profile_kernels(output_dir, device, configurations,
                                           args.warmup, args.repeat, args.seed)
                for row, (symbol, kernel_samples) in zip(results, profiles):
                    row["kernel"] = symbol
                    row.update({f"kernel_{key}": value
                                for key, value in latency_statistics(kernel_samples).items()})
                    samples.extend({"case": row["case"], "op": row["op"], "mapped": row["mapped"],
                                    "metric": "kernel", "sample": i, "duration_us": value}
                                   for i, value in enumerate(kernel_samples))
                    print(f"KERNEL {row['case']} {row['op']:8} map={row['mapped']} "
                          f"{row['kernel_median_us']:.3f} us", flush=True)
        summary["status"] = "passed"
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for filename, rows in (("performance.csv", results), ("samples.csv", samples)):
            if rows:
                with (output_dir / filename).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(
                        key for row in rows for key in row)))
                    writer.writeheader()
                    writer.writerows(rows)
    print(f"PASS: {len(results)} configurations. {output_dir.resolve() / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
