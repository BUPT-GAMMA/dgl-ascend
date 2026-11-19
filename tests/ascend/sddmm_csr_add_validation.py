"""Accuracy, all-key coverage and performance of native Ascend CSR SDDMM add.

Run after building DGL with USE_ASCEND=ON:
    PYTHONPATH=python python3 tests/ascend/sddmm_csr_add_validation.py

The accuracy table retains all 19 submitted entries, including its duplicate.
Both edge-map modes are tested for every entry. A separate matrix verifies all
36 semantic keys across the four index/feature type combinations using actual
device kernel names. Every output element is compared with a CPU reference.

Performance uses the four submitted int32/float32 cases in both edge-map modes.
Synchronized gsddmm API times and profiler device-kernel times are collected in
separate runs. Graph construction, input generation and H2D precede timing.
Each run writes CSV results, raw profiler data and a JSON summary to its own
timestamped directory below --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import statistics
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_npu

# Register Ascend tensor support before loading libdgl.so.
import dgl


TARGETS = ("u", "e", "v")
TARGET_CODES = {target: index for index, target in enumerate(TARGETS)}
ID_TYPES = (torch.int32, torch.int64)
FEATURE_TYPES = (torch.float32, torch.float64)
ETYPE = ("source", "edge", "destination")


@dataclass(frozen=True)
class Case:
    name: str
    targets: tuple[str, str]
    graph: tuple[int, int, int]
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]


ACCURACY_CASES = (
    Case("accuracy_01_eu", ("e", "u"), (128, 96, 511), (32,), (32,)),
    Case("accuracy_02_ee", ("e", "e"), (1024, 1024, 8192), (64,), (64,)),
    Case("accuracy_03_ev", ("e", "v"), (4096, 2048, 65536), (64,), (64,)),
    Case("accuracy_04_vu", ("v", "u"), (10000, 5000, 200000), (128,), (128,)),
    Case("accuracy_05_ve", ("v", "e"), (50000, 20000, 800000), (64,), (64,)),
    Case("accuracy_06_vv", ("v", "v"), (100000, 100000, 2000000), (64,), (64,)),
    Case("accuracy_07_uv", ("u", "v"), (4096, 4096, 65536), (8, 32), (8, 32)),
    Case("accuracy_08_uv", ("u", "v"), (50000, 50000, 800000), (8, 16), (8, 16)),
    Case("accuracy_09_uv", ("u", "v"), (10000, 20000, 200000), (128,), (128,)),
    Case("accuracy_10_uv", ("u", "v"), (100000, 100000, 1000000), (64,), (64,)),
    Case("accuracy_11_uv", ("u", "v"), (8192, 8192, 65536), (128,), (128,)),
    Case("accuracy_12_ue", ("u", "e"), (10000, 10000, 500000), (64,), (64,)),
    Case("accuracy_13_uv", ("u", "v"), (12000, 2000, 60000), (4, 32), (4, 32)),
    Case("accuracy_14_uv", ("u", "v"), (50000, 5000, 500000), (8, 16), (8, 16)),
    Case("accuracy_15_uv", ("u", "v"), (200000, 200000, 1000000), (64,), (64,)),
    Case("accuracy_16_uv", ("u", "v"), (8192, 8192, 131072), (4, 64), (4, 64)),
    # The submitted table includes this layout twice, so both rows are exercised.
    Case("accuracy_17_uv", ("u", "v"), (4096, 4096, 65536), (8, 32), (8, 32)),
    Case("accuracy_18_uv", ("u", "v"), (50000, 50000, 1000000), (4, 32), (4, 32)),
    Case("accuracy_19_uv", ("u", "v"), (20000, 20000, 250000), (96,), (96,)),
)

PERFORMANCE_CASES = (
    ACCURACY_CASES[2], ACCURACY_CASES[3], ACCURACY_CASES[11], ACCURACY_CASES[13],
)

ACCURACY_FIELDS = (
    "group", "case", "M", "N", "E", "lhs_target", "rhs_target",
    "lhs_shape", "rhs_shape", "out_shape", "idtype", "dtype", "key",
    "has_edge_map", "use_broadcast", "expected_kernel", "status",
    "max_abs_error", "compared_elements", "error",
)


@dataclass
class PreparedCase:
    case: Case
    idtype: torch.dtype
    dtype: torch.dtype
    mapped: bool
    graph: dgl.DGLGraph
    lhs_cpu: torch.Tensor
    rhs_cpu: torch.Tensor
    lhs_npu: torch.Tensor
    rhs_npu: torch.Tensor
    row_ids: torch.Tensor
    indices: torch.Tensor
    edge_ids: torch.Tensor
    key: int
    kernel_name: str


def case_metadata(case, idtype, dtype, mapped, group="accuracy"):
    """Describe one invocation using the host dispatch key convention."""
    source_nodes, destination_nodes, edges = case.graph
    sizes = {"u": source_nodes, "e": edges, "v": destination_nodes}
    lhs_target, rhs_target = case.targets
    use_broadcast = case.lhs_shape != case.rhs_shape
    key = ((TARGET_CODES[lhs_target] * 3 + TARGET_CODES[rhs_target]) * 2
           + int(use_broadcast)) * 2 + int(mapped)
    dtype_bits = torch.finfo(dtype).bits
    id_bits = torch.iinfo(idtype).bits
    output_features = tuple(torch.broadcast_shapes(case.lhs_shape, case.rhs_shape))
    return {
        "group": group,
        "case": case.name,
        "M": source_nodes,
        "N": destination_nodes,
        "E": edges,
        "lhs_target": lhs_target,
        "rhs_target": rhs_target,
        "lhs_shape": json.dumps((sizes[lhs_target],) + case.lhs_shape),
        "rhs_shape": json.dumps((sizes[rhs_target],) + case.rhs_shape),
        "out_shape": json.dumps((edges,) + output_features),
        "idtype": str(idtype).removeprefix("torch."),
        "dtype": str(dtype).removeprefix("torch."),
        "key": key,
        "has_edge_map": int(mapped),
        "use_broadcast": int(use_broadcast),
        "expected_kernel": f"sddmm_csr_add_f{dtype_bits}_i{id_bits}_{key}",
    }


def prepare_case(case, idtype, dtype, mapped, device, seed):
    """Construct an exact-size random CSR graph and reproducible input tensors."""
    source_nodes, destination_nodes, edges = case.graph
    generator = torch.Generator().manual_seed(seed)
    sampled_rows = torch.randint(source_nodes, (edges,), generator=generator)
    counts = torch.bincount(sampled_rows, minlength=source_nodes)
    row_ids = torch.repeat_interleave(torch.arange(source_nodes), counts)
    indptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    indices = torch.randint(destination_nodes, (edges,), generator=generator)
    edge_ids = torch.randperm(edges, generator=generator) if mapped else torch.arange(edges)
    if mapped and edges > 1 and torch.equal(edge_ids, torch.arange(edges)):
        edge_ids = edge_ids.roll(1)
    data = edge_ids.to(idtype) if mapped else torch.empty(0, dtype=idtype)
    graph = dgl.heterograph(
        {ETYPE: ("csr", (indptr.to(idtype), indices.to(idtype), data))},
        num_nodes_dict={"source": source_nodes, "destination": destination_nodes},
        idtype=idtype,
    ).formats("csr").to(device)
    actual_indptr, actual_indices, actual_data = graph.adj_tensors("csr")
    assert graph.formats()["created"] == ["csr"]
    assert actual_indptr.dtype == idtype and actual_indices.dtype == idtype
    assert actual_data.numel() == (edges if mapped else 0)
    assert graph.num_nodes("source") == source_nodes
    assert graph.num_nodes("destination") == destination_nodes
    assert graph.num_edges() == edges

    feature_generator = torch.Generator().manual_seed(seed + 1)
    sizes = {"u": source_nodes, "e": edges, "v": destination_nodes}
    lhs_cpu = torch.randn((sizes[case.targets[0]],) + case.lhs_shape,
                          dtype=dtype, generator=feature_generator)
    rhs_cpu = torch.randn((sizes[case.targets[1]],) + case.rhs_shape,
                          dtype=dtype, generator=feature_generator)
    metadata = case_metadata(case, idtype, dtype, mapped)
    return PreparedCase(
        case=case, idtype=idtype, dtype=dtype, mapped=mapped, graph=graph,
        lhs_cpu=lhs_cpu, rhs_cpu=rhs_cpu,
        lhs_npu=lhs_cpu.to(device), rhs_npu=rhs_cpu.to(device),
        row_ids=row_ids, indices=indices, edge_ids=edge_ids,
        key=metadata["key"], kernel_name=metadata["expected_kernel"],
    )


def run_add(prepared):
    return dgl.ops.gsddmm(
        prepared.graph, "add", prepared.lhs_npu, prepared.rhs_npu,
        lhs_target=prepared.case.targets[0], rhs_target=prepared.case.targets[1],
    )


def check_accuracy(prepared, rtol=0.0, atol=0.0):
    """Compare every output element against an independent CPU CSR reference."""
    metrics = {"status": "PASS", "max_abs_error": 0.0,
               "compared_elements": 0, "error": ""}
    case = prepared.case
    edges = case.graph[2]
    feature_shape = tuple(torch.broadcast_shapes(case.lhs_shape, case.rhs_shape))
    expected_shape = (edges,) + feature_shape
    actual = run_add(prepared).cpu()
    if tuple(actual.shape) != expected_shape or actual.dtype != prepared.dtype:
        metrics.update(status="FAIL", error=(
            f"Expected shape/dtype {expected_shape}/{prepared.dtype}, "
            f"received {tuple(actual.shape)}/{actual.dtype}"))
        return metrics
    if prepared.graph.formats()["created"] != ["csr"]:
        metrics.update(status="FAIL", error="Graph acquired an additional sparse format")
        return metrics

    row_indices = {"u": prepared.row_ids, "e": prepared.edge_ids, "v": prepared.indices}
    chunk_edges = max(1, (1 << 20) // math.prod(feature_shape))
    feature_rank = len(feature_shape)
    for begin in range(0, edges, chunk_edges):
        end = min(edges, begin + chunk_edges)
        lhs = prepared.lhs_cpu[row_indices[case.targets[0]][begin:end]]
        rhs = prepared.rhs_cpu[row_indices[case.targets[1]][begin:end]]
        lhs = lhs.reshape((end - begin,) + (1,) * (feature_rank - len(case.lhs_shape))
                          + case.lhs_shape)
        rhs = rhs.reshape((end - begin,) + (1,) * (feature_rank - len(case.rhs_shape))
                          + case.rhs_shape)
        expected = lhs + rhs
        result = actual[prepared.edge_ids[begin:end]]
        difference = (result - expected).abs()
        chunk_error = difference.max().item()
        metrics["max_abs_error"] = max(metrics["max_abs_error"],
                                       float("inf") if math.isnan(chunk_error) else chunk_error)
        metrics["compared_elements"] += expected.numel()
        try:
            torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
        except AssertionError as error:
            metrics["status"] = "FAIL"
            if not metrics["error"]:
                metrics["error"] = f"CSR edge positions [{begin}, {end}): {error}"
    assert metrics["compared_elements"] == edges * math.prod(feature_shape)
    return metrics


def _run_accuracy_cases(cases, csv_path, group, device, seed):
    """Write each completed invocation before advancing to the next input."""
    passed = 0
    compared_elements = 0
    max_abs_error = 0.0
    covered_keys = {}
    expected_kernels = set()
    total_cases = len(cases) * len(ID_TYPES) * len(FEATURE_TYPES) * 2
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACCURACY_FIELDS)
        writer.writeheader()
        handle.flush()
        for case_index, case in enumerate(cases):
            for idtype, dtype, mapped in itertools.product(ID_TYPES, FEATURE_TYPES, (False, True)):
                row = case_metadata(case, idtype, dtype, mapped, group)
                try:
                    prepared = prepare_case(case, idtype, dtype, mapped, device, seed + case_index)
                    metrics = check_accuracy(prepared)
                    del prepared
                except Exception as error:
                    row.update(status="FAIL", max_abs_error="", compared_elements=0,
                               error=f"{type(error).__name__}: {error}")
                    writer.writerow(row)
                    handle.flush()
                    raise
                row.update(metrics)
                writer.writerow(row)
                handle.flush()
                if metrics["status"] != "PASS":
                    raise AssertionError(f"{case.name}, {idtype}, {dtype}, map={mapped}: {metrics['error']}")
                passed += 1
                compared_elements += metrics["compared_elements"]
                max_abs_error = max(max_abs_error, metrics["max_abs_error"])
                type_name = f"{row['idtype']}/{row['dtype']}"
                covered_keys.setdefault(type_name, set()).add(row["key"])
                expected_kernels.add(row["expected_kernel"])
                print(f"[{group} {passed}/{total_cases}] {case.name} "
                      f"{type_name} map={int(mapped)} key={row['key']} PASS "
                      f"max_abs_error={metrics['max_abs_error']}", flush=True)
    return {
        "passed": passed,
        "compared_elements": compared_elements,
        "max_abs_error": max_abs_error,
        "covered_keys": {name: sorted(keys) for name, keys in covered_keys.items()},
        "expected_kernels": sorted(expected_kernels),
        "csv": str(csv_path),
    }


def run_accuracy(output_dir, device, seed):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return _run_accuracy_cases(ACCURACY_CASES, output_dir / "accuracy.csv",
                               "accuracy", device, seed)


def make_profiler(output_dir):
    """Capture one block of NPU work and export its Level1 task records."""
    from torch_npu.profiler import (
        ProfilerAction, ProfilerActivity, ProfilerLevel, profile,
        tensorboard_trace_handler,
    )
    from torch_npu.profiler.experimental_config import AiCMetrics, _ExperimentalConfig

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        schedule=lambda step: ProfilerAction.RECORD_AND_SAVE if step == 0 else ProfilerAction.NONE,
        on_trace_ready=tensorboard_trace_handler(str(output_dir)),
        experimental_config=_ExperimentalConfig(
            profiler_level=ProfilerLevel.Level1, aic_metrics=AiCMetrics.AiCoreNone),
    )


def read_kernel_durations(profile_dir):
    """Read SDDMMCsrAdd device task durations in microseconds by exact symbol."""
    tasks = []
    for path in sorted(Path(profile_dir).glob("**/ASCEND_PROFILER_OUTPUT/kernel_details.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                name = row["Name"].strip()
                if name.startswith("sddmm_csr_add_"):
                    tasks.append((float(row["Start Time(us)"].strip()), name,
                                  float(row["Duration(us)"].strip())))
    durations = {}
    for _, name, duration in sorted(tasks, key=lambda task: task[0]):
        durations.setdefault(name, []).append(duration)
    return durations


def run_key_coverage(output_dir, device, seed):
    """Validate all keys and confirm each specialization in the device trace."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_root = output_dir / "key_coverage_profile"
    profile_root.mkdir(exist_ok=True)
    profile_dir = Path(tempfile.mkdtemp(prefix="run_", dir=profile_root))
    cases = tuple(
        Case(f"key_{lhs}{rhs}_broadcast_{int(broadcast)}", (lhs, rhs), (17, 13, 73),
             (4, 1) if broadcast else (32,), (1, 8) if broadcast else (32,))
        for lhs, rhs, broadcast in itertools.product(TARGETS, TARGETS, (False, True))
    )
    torch.npu.synchronize()
    with make_profiler(profile_dir) as profiler:
        summary = _run_accuracy_cases(cases, output_dir / "key_accuracy.csv",
                                      "key_coverage", device, seed)
        torch.npu.synchronize()
        profiler.step()
    for type_name, keys in summary["covered_keys"].items():
        assert keys == list(range(36)), f"Incomplete keys for {type_name}: {keys}"
    assert len(summary["covered_keys"]) == 4
    expected = set(summary["expected_kernels"])
    assert len(expected) == 144
    profiler_csvs = sorted(profile_dir.glob("**/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    kernel_counts = Counter({name: len(values) for name, values in read_kernel_durations(profile_dir).items()})
    coverage_csv = output_dir / "kernel_coverage.csv"
    with coverage_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("kernel", "observed_launches", "status"))
        writer.writeheader()
        for name in sorted(expected):
            writer.writerow({"kernel": name, "observed_launches": kernel_counts[name],
                             "status": "PASS" if kernel_counts[name] else "FAIL"})
        handle.flush()
    observed = set(kernel_counts)
    summary.update(
        expected_kernel_count=len(expected), observed_kernel_count=len(observed),
        observed_kernels=sorted(observed), kernel_launch_counts=dict(sorted(kernel_counts.items())),
        kernel_coverage_csv=str(coverage_csv), profile_dir=str(profile_dir),
        profiler_csvs=[str(path) for path in profiler_csvs],
    )
    assert observed == expected, (
        f"Device kernel coverage mismatch, missing={sorted(expected - observed)}, "
        f"unexpected={sorted(observed - expected)}")
    return summary
def latency_statistics(samples):
    ordered = sorted(samples)
    return {
        'mean_us': statistics.mean(samples),
        'median_us': statistics.median(samples),
        'p90_us': ordered[math.ceil(0.9 * len(ordered)) - 1],
        'min_us': ordered[0],
        'max_us': ordered[-1],
    }


def run_performance(output_dir, device, seed, warmup, repeat):
    prepared_cases = []
    api_timings = {}
    for case_index, case in enumerate(PERFORMANCE_CASES):
        for mapped in (False, True):
            prepared = prepare_case(
                case, torch.int32, torch.float32, mapped, device,
                seed + case_index,
            )
            accuracy = check_accuracy(prepared)
            if accuracy["status"] != "PASS":
                raise AssertionError(f"{case.name}, map={mapped}: {accuracy['error']}")
            for _ in range(warmup):
                run_add(prepared)
            torch.npu.synchronize()
            samples = []
            for _ in range(repeat):
                begin = time.perf_counter_ns()
                output = run_add(prepared)
                samples.append((time.perf_counter_ns() - begin) / 1000)
                del output
            api_timings[prepared.kernel_name] = samples
            prepared_cases.append(prepared)
            print(
                f'API {case.name} map={mapped} '
                f'median={statistics.median(samples):.3f} us', flush=True,
            )

    profile_dir = output_dir / 'performance_profile'
    with make_profiler(profile_dir) as profiler:
        for prepared in prepared_cases:
            # The first warmup records for each symbol are excluded below.
            for _ in range(warmup + repeat):
                run_add(prepared)
        torch.npu.synchronize()
        profiler.step()
    durations = read_kernel_durations(profile_dir)
    results = []
    sample_rows = []
    for prepared in prepared_cases:
        name = prepared.kernel_name
        recorded = durations.get(name, [])
        if len(recorded) != warmup + repeat:
            raise AssertionError(
                f'{name}: expected {warmup + repeat} profiler records, '
                f'got {len(recorded)}'
            )
        kernel_samples = recorded[warmup:]
        case = prepared.case
        row = {
            'case': case.name,
            'lhs_target': case.targets[0],
            'rhs_target': case.targets[1],
            'M': case.graph[0], 'N': case.graph[1], 'E': case.graph[2],
            'lhs_shape': str(tuple(prepared.lhs_npu.shape)),
            'rhs_shape': str(tuple(prepared.rhs_npu.shape)),
            'out_shape': str((case.graph[2],) + tuple(torch.broadcast_shapes(case.lhs_shape, case.rhs_shape))),
            'idtype': 'int32', 'dtype': 'float32',
            'mapped': prepared.mapped,
            'key': prepared.key, 'kernel': name,
            'warmup': warmup, 'repeat': repeat,
        }
        row.update({f'kernel_{key}': value for key, value in latency_statistics(kernel_samples).items()})
        row.update({f'api_{key}': value for key, value in latency_statistics(api_timings[name]).items()})
        results.append(row)
        for index, (api_us, kernel_us) in enumerate(zip(api_timings[name], kernel_samples)):
            for metric, duration in (('api', api_us), ('kernel', kernel_us)):
                sample_rows.append({
                    'case': case.name, 'mapped': prepared.mapped,
                    'kernel': name, 'sample': index + 1,
                    'metric': metric, 'duration_us': duration,
                })
        print(
            f'KERNEL {case.name} map={prepared.mapped} '
            f'median={row["kernel_median_us"]:.3f} us', flush=True,
        )
    for filename, rows in (('performance.csv', results), ('performance_samples.csv', sample_rows)):
        with (output_dir / filename).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {
        'status': 'passed', 'case_count': len(PERFORMANCE_CASES),
        'configuration_count': len(results),
        'api_timing': 'perf_counter_ns around synchronized gsddmm, profiler disabled; includes output allocation, host dispatch and tiling transfers',
        'kernel_timing': 'CANN Level1 kernel_details.csv Duration(us); separate run from API timing',
        'samples_are_paired': False,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('all', 'accuracy', 'keys', 'performance'), default='all')
    parser.add_argument('--output-dir', type=Path, default=Path('build/sddmm_csr_add_validation'))
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1 or args.threads < 1:
        parser.error('warmup must be nonnegative; repeat and threads must be positive')
    if not torch.npu.is_available():
        raise RuntimeError('Ascend NPU is required')
    torch.set_num_threads(args.threads)
    torch.npu.set_device(args.device)
    device = torch.device(f'npu:{args.device}')
    run_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True)
    summary = {
        'status': 'running', 'output_dir': str(output_dir.resolve()),
        'started_utc': run_name, 'mode': args.mode,
        'seed': args.seed, 'warmup': args.warmup, 'repeat': args.repeat,
        'threads': args.threads,
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
            'dgl': dgl.__version__,
            'device': str(device), 'device_name': torch.npu.get_device_name(args.device),
        },
        'operation': 'SDDMMCsrAdd',
        'accuracy_comparison': 'full-output CPU reference, rtol=0, atol=0',
        'edge_map_modes': [False, True],
        'table_entries': 19, 'unique_table_layouts': 18,
    }
    print(f'Results directory: {output_dir.resolve()}', flush=True)
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    try:
        with torch.inference_mode():
            if args.mode in ('all', 'accuracy', 'keys'):
                summary['key_coverage'] = run_key_coverage(output_dir, device, args.seed)
                summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
            if args.mode in ('all', 'accuracy'):
                summary['accuracy'] = run_accuracy(output_dir, device, args.seed)
                summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
            if args.mode in ('all', 'performance'):
                summary['performance'] = run_performance(output_dir, device, args.seed, args.warmup, args.repeat)
        summary['status'] = 'passed'
    except Exception as error:
        summary['status'] = 'failed'
        summary['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'PASS. Summary: {summary_path.resolve()}', flush=True)


if __name__ == '__main__':
    main()
