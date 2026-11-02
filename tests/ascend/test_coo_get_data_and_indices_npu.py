#!/usr/bin/env python3
"""COO-only Python/NPU test, independent of the full libdgl.so.

Uses current repository COO Host/Kernel and native CPU reference, comparing all
three outputs in order. Does not import dgl or build other array operators.
Requires an existing Ascend CMake configuration and the compiled COO kernel.
Run from the repository root:
  python3 tests/ascend/test_coo_get_data_and_indices_npu.py --build build_coo_pr

This is a Python ctypes direct-call test, not a dgl.graph integration test.
The embedded C++ bridge is generated only inside the selected build directory.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


BRIDGE_SOURCE = r'''#include <dgl/array.h>

#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
using dgl::aten::COOMatrix;
using dgl::runtime::NDArray;

template <typename T>
NDArray Input(const void* pointer, int64_t count) {
  if (count < 0 || static_cast<uint64_t>(count) >
          std::numeric_limits<size_t>::max() / sizeof(T)) {
    throw std::invalid_argument("Invalid input length");
  }
  std::vector<T> values(static_cast<size_t>(count));
  if (count != 0) {
    if (!pointer) throw std::invalid_argument("Missing input buffer");
    std::memcpy(values.data(), pointer, values.size() * sizeof(T));
  }
  return NDArray::FromVector(values);
}

template <typename T>
void Query(
    int use_npu, int64_t num_rows, int64_t num_cols,
    int64_t nnz, const void* coo_rows, const void* coo_cols,
    const void* data, int64_t row_count, const void* rows,
    int64_t col_count, const void* cols, int64_t capacity,
    void* out_rows, void* out_cols, void* out_data, int64_t* out_count) {
  if (!out_count || capacity < 0 || num_rows < 0 || num_cols < 0) {
    throw std::invalid_argument("Invalid output buffer or dimensions");
  }
  COOMatrix coo{
      num_rows, num_cols, Input<T>(coo_rows, nnz), Input<T>(coo_cols, nnz),
      data ? Input<T>(data, nnz) : dgl::aten::NullArray()};
  auto query_rows = Input<T>(rows, row_count);
  auto query_cols = Input<T>(cols, col_count);
  const DGLContext ctx{use_npu ? kDGLAscend : kDGLCPU, 0};
  if (use_npu) {
    coo = coo.CopyTo(ctx);
    query_rows = query_rows.CopyTo(ctx);
    query_cols = query_cols.CopyTo(ctx);
  }
  // The linked definition is extracted unchanged from the current array.cc.
  auto result = dgl::aten::COOGetDataAndIndices(coo, query_rows, query_cols);
  const int64_t count = result.at(0)->shape[0];
  if (count > capacity) throw std::length_error("Output capacity exceeded");
  void* outputs[] = {out_rows, out_cols, out_data};
  for (size_t i = 0; i < 3; ++i) {
    if (result.at(i)->shape[0] != count) {
      throw std::runtime_error("Output lengths differ");
    }
    if (count != 0) {
      if (!outputs[i]) throw std::invalid_argument("Missing output buffer");
      auto host = result[i].CopyTo(DGLContext{kDGLCPU, 0});
      std::memcpy(outputs[i], host->data, static_cast<size_t>(count) * sizeof(T));
    }
  }
  *out_count = count;
}
}  // namespace

extern "C" const char* coo_python_last_error() {
  return last_error.c_str();
}

extern "C" int coo_python_query(
    int bits, int use_npu, int64_t num_rows, int64_t num_cols,
    int64_t nnz, const void* coo_rows, const void* coo_cols,
    const void* data, int64_t row_count, const void* rows,
    int64_t col_count, const void* cols, int64_t capacity,
    void* out_rows, void* out_cols, void* out_data, int64_t* out_count) {
  try {
    last_error.clear();
    if (out_count) *out_count = 0;
    if (bits == 32) {
      Query<int32_t>(use_npu, num_rows, num_cols, nnz, coo_rows, coo_cols,
          data, row_count, rows, col_count, cols, capacity,
          out_rows, out_cols, out_data, out_count);
    } else if (bits == 64) {
      Query<int64_t>(use_npu, num_rows, num_cols, nnz, coo_rows, coo_cols,
          data, row_count, rows, col_count, cols, capacity,
          out_rows, out_cols, out_data, out_count);
    } else {
      throw std::invalid_argument("Expected int32 or int64");
    }
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return 1;
  } catch (...) {
    last_error = "Unknown C++ exception";
    return 1;
  }
}
'''

def extract_function(path, signature, template=False):
    text = path.read_text(encoding="utf-8-sig")
    if text.count(signature) != 1:
        raise RuntimeError(f"Expected exactly one function in {path}")
    start = text.index(signature)
    if template:
        prefix = "template <DGLDeviceType XPU, typename IdType>\n"
        if not text[:start].endswith(prefix):
            raise RuntimeError(f"Unexpected template declaration in {path}")
        start -= len(prefix)
    end = text.index("\n}", start) + 2
    return text[start:end] + "\n"

def generate(repo, out):
    signature = "std::vector<NDArray> COOGetDataAndIndices(\n"
    public = extract_function(repo / "src/array/array.cc", signature)
    if "kDGLAscend, IdType" not in public:
        raise RuntimeError("array.cc still lacks the Ascend dispatch")
    cpu = extract_function(
        repo / "src/array/cpu/spmat_op_impl_coo.cc", signature, template=True)
    includes = '#include <dgl/array.h>\n#include <algorithm>\n#include <unordered_map>\n'
    includes += '#include "src/array/array_op.h"\n'
    includes += '#include "src/array/cpu/array_utils.h"\n'
    code = includes + "namespace dgl { namespace aten {\nnamespace impl {\n" + cpu
    for dtype in ("int32_t", "int64_t"):
        code += (f"template std::vector<NDArray> COOGetDataAndIndices<kDGLCPU, {dtype}>("
                 "COOMatrix, NDArray, NDArray);\n")
    code += "}  // namespace impl\n" + public + "}}  // namespace dgl::aten\n"
    out.mkdir(parents=True, exist_ok=True)
    source = out / "coo_dispatch_cpu.cc"
    source.write_text(code, encoding="utf-8", newline="\n")
    (out / "extracted_functions.json").write_text(json.dumps({
        "public_function_sha256": hashlib.sha256(public.encode()).hexdigest(),
        "cpu_function_sha256": hashlib.sha256(cpu.encode()).hexdigest(),
        "scope": "Extracted current functions, not the full DGL shared library",
    }, indent=2), encoding="utf-8")
    return source

def configure(path):
    library = C.CDLL(str(path))
    library.coo_python_query.restype = C.c_int
    library.coo_python_query.argtypes = [
        C.c_int, C.c_int, C.c_int64, C.c_int64, C.c_int64,
        C.c_void_p, C.c_void_p, C.c_void_p, C.c_int64, C.c_void_p,
        C.c_int64, C.c_void_p, C.c_int64,
        C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_int64)]
    library.coo_python_last_error.restype = C.c_char_p
    library.coo_python_last_error.argtypes = []
    return library

def query(library, bits, use_npu, shape, coo_rows, coo_cols, data, rows, cols):
    if len(coo_rows) != len(coo_cols) or (data is not None and len(data) != len(coo_rows)):
        raise ValueError("COO input lengths differ")
    scalar = C.c_int32 if bits == 32 else C.c_int64

    def array(values):
        return (scalar * len(values))(*values)

    # Keep all input buffers alive until the synchronous bridge returns.
    inputs = [array(values) if values is not None else None
              for values in (coo_rows, coo_cols, data, rows, cols)]
    capacity = len(coo_rows) * max(len(rows), len(cols))
    outputs = [(scalar * max(capacity, 1))() for _ in range(3)]
    count = C.c_int64()
    status = library.coo_python_query(
        bits, int(use_npu), *shape, len(coo_rows), *inputs[:3],
        len(rows), inputs[3], len(cols), inputs[4], capacity,
        *outputs, C.byref(count))
    if status:
        raise RuntimeError(library.coo_python_last_error().decode("utf-8", "replace"))
    return tuple(list(output[:count.value]) for output in outputs)

def run_cases(library_path):
    library = configure(Path(library_path).resolve())
    cases = [
        ("duplicates", (3, 3), [1, 0, 1, 1], [2, 1, 2, 2], [10, 20, 30, 40], [1], [2]),
        ("implicit", (3, 3), [1, 0, 1, 1], [2, 1, 2, 2], None, [1, 0], [2, 1]),
        ("rows_broadcast", (3, 3), [1, 1, 1], [0, 1, 2], None, [1], [0, 1, 2]),
        ("cols_broadcast", (3, 3), [0, 1, 2], [2, 2, 2], None, [0, 1, 2], [2]),
        ("q200", (3, 3), [1, 0, 1, 1], [2, 1, 2, 2], [10, 20, 30, 40], [1] * 200, [2]),
        ("no_match", (3, 3), [1], [2], None, [0], [0]),
        ("empty_coo", (3, 3), [], [], None, [1], [2]),
        ("empty_query", (3, 3), [1], [2], None, [], []),
        ("tile257", (3, 3), [1] * 257, [2] * 257, list(range(257)), [1], [2]),
    ]
    passed = 0
    for bits in (32, 64):
        large = 2 ** 24 + 1 if bits == 32 else 2 ** 40 + 1
        extra = ("large_integer", (large + 2, large + 2),
                 [large, large + 1], [large, large], [7, 8], [large], [large])
        for name, *inputs in [*cases, extra]:
            expected = query(library, bits, False, *inputs)
            actual = query(library, bits, True, *inputs)
            for label, cpu, npu in zip(("ret_rows", "ret_cols", "ret_data"), expected, actual):
                if cpu != npu:
                    raise AssertionError(f"int{bits}/{name}/{label}: CPU={cpu[:12]}, NPU={npu[:12]}")
            if name == "duplicates" and expected != ([1, 1, 1], [2, 2, 2], [10, 30, 40]):
                raise AssertionError("Unexpected native CPU result")
            passed += 1
            print(f"PASS int{bits} {name}: {len(actual[0])} results", flush=True)
    print(f"Python COO-only: {passed}/{passed} passed (not a dgl.graph API test)")

def build_and_test(repo, build):
    repo = repo.resolve()
    build = build.resolve()
    out = build / "coo_python_only"
    out.mkdir(parents=True, exist_ok=True)
    generated = generate(repo, out)
    bridge_source = out / "coo_python_bridge.cc"
    bridge_source.write_text(BRIDGE_SOURCE, encoding="utf-8", newline="\n")
    flags_text = (build / "CMakeFiles/dgl.dir/flags.make").read_text()
    flags = []
    for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        match = re.search(r"^" + key + r" = (.*)$", flags_text, re.M)
        if not match:
            raise RuntimeError(f"Missing {key}")
        flags.extend(shlex.split(match.group(1)))
    if "-DDGL_USE_ASCEND" not in flags or "-DDGL_USE_CUDA" in flags:
        raise RuntimeError("Requires USE_ASCEND=ON, USE_CUDA=OFF")
    dgl_link = shlex.split((build / "CMakeFiles/dgl.dir/link.txt").read_text())
    test_link = shlex.split((build / "CMakeFiles/runUnitTests.dir/link.txt").read_text())
    kernel_name = "libdgl_ascendc_coo_get_data_and_indices_kernel.a"
    if not (build / "lib" / kernel_name).is_file():
        raise RuntimeError("Compile ONLY dgl_ascendc_coo_get_data_and_indices_kernel first")
    # Retain known-working external runtime dependencies from the server config.
    # Remove libdgl, every other kernel, and all GTest files/libraries.
    libraries = []
    skip = False
    for token in test_link[1:]:
        if skip:
            skip = False
            continue
        if token == "-o":
            skip = True
        elif token.endswith(".o") or Path(token).name == "libdgl.so" or "libgtest" in token:
            continue
        else:
            libraries.append(token)
    first_lib = next(i for i, token in enumerate(dgl_link) if token.startswith("-L"))
    for token in dgl_link[first_lib:]:
        if "libdgl_ascendc_" in token and Path(token).name != kernel_name:
            continue
        libraries.append(token)
    libraries = [t for t in libraries
                 if not (t.startswith("-L") and ("/stub" in t or "/simulator/" in t))]
    for token in libraries:
        if token.endswith(".a") and not (build / token).is_file():
            raise RuntimeError(f"Missing existing support library: {token}")
    runtime_sources = [p for p in sorted((repo / "src/runtime").glob("*.cc"))
                       if p.name != "utils.cc"]
    sources = [generated, bridge_source,
               repo / "src/array/ascend/coo_get_data_and_indices.cc",
               repo / "src/runtime/ascend/ascend_device_api.cc",
               repo / "src/c_api_common.cc",
               repo / "src/graph/serialize/zerocopy_serializer.cc", *runtime_sources]
    objects = []
    for index, source in enumerate(sources):
        obj = out / f"{index:02d}_{source.name}.o"
        print(f"Compile {source.name}", flush=True)
        subprocess.run([dgl_link[0], *flags, "-std=c++17", "-O2", "-fPIC",
                        "-ffunction-sections", "-fdata-sections", "-I" + str(repo),
                        "-c", str(source), "-o", str(obj)], cwd=build, check=True)
        objects.append(str(obj))
    # Keep the small bridge private from any installed DGL: export only two C APIs.
    exports = out / "exports.map"
    exports.write_text("COO_PYTHON_ONLY_1 { global: coo_python_query; coo_python_last_error; local: *; };\n")
    library = out / "libcoo_python_only.so"
    print("Link COO-only library (no libdgl, UnaryElewise or SpMM)", flush=True)
    subprocess.run([dgl_link[0], "-shared", *objects, "-o", str(library),
                    "-Wl,--gc-sections,--no-undefined", "-Wl,--version-script=" + str(exports),
                    *libraries, "-pthread"], cwd=build, check=True)
    env = os.environ.copy()
    directories = []
    for token in libraries:
        if token.startswith("-L"):
            directory = token[2:]
        elif token.startswith("/") and ".so" in token:
            directory = str(Path(token).parent)
        else:
            continue
        if "/stub" not in directory and "/simulator/" not in directory:
            directories.append(directory)
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(directories)) + ":" + env.get("LD_LIBRARY_PATH", "")
    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--library", str(library)], cwd=build, env=env, check=True)

def default_paths():
    repo = Path(__file__).resolve().parents[2]
    build = Path(os.environ.get("DGL_COO_TEST_BUILD", str(repo / "build")))
    return repo, build


def test_coo_python_only():
    """Pytest entry: compilation and 20 real Python CPU/NPU comparisons."""
    repo, build = default_paths()
    build_and_test(repo, build)


def main():
    default_repo, _ = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=default_repo)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--library", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.library:
        run_cases(args.library)
        return
    repo = args.repo.resolve()
    build = args.build or Path(os.environ.get("DGL_COO_TEST_BUILD", str(repo / "build")))
    build_and_test(repo, build)


if __name__ == "__main__":
    main()
