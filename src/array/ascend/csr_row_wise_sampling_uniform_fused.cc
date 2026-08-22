/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_sampling_uniform_fused.cc
 * @brief Ascend host launcher for uniform CSR row-wise fused sampling.
 *
 * Multi-core design (twin of csr_row_wise_sampling_uniform.cc per
 * ADR-0011): the host computes per-row pick counts from row degrees (one
 * CSRGetRowNNZ launch + one D2H copy), builds nnz-balanced row-range
 * partitions plus per-block output offsets as prefix sums, allocates all
 * outputs exactly, and launches the AIV kernel. Blocks write disjoint
 * output ranges, so the kernel needs no cross-block reduction.
 *
 * Fused extras:
 * - Returns pair<CSRMatrix, IdArray>: the block-local CSR (indptr/col/idx)
 *   plus picked_coo_rows carrying the ROW POSITION of each picked edge.
 * - seed_mapping: the kernel emits a compact interleaved (rid, pos) pair
 *   array (ADR-0014 — AscendC has no GM scatter primitive, so the kernel
 *   only produces contiguous batch writes). This launcher scatters the
 *   pairs into seed_mapping on the host after the kernel completes
 *   (A1 closure: one D2H of the pairs + one H2D of the mapping).
 * - new_seed_nodes: host vector filled with a copy of rows (CPU
 *   reference semantics, CSRRowWisePickFused).
 */

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#define ASCEND_CALL(func)                                   \
  {                                                         \
    aclError e = (func);                                    \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_uniform_fused_int32(
    uint32_t blockDim, aclrtStream stream, void* indptr, void* indices,
    void* data, void* rows, void* out_ptr, void* out_rows, void* out_cols,
    void* out_idxs, void* seed_pairs, void* row_split, void* out_starts,
    void* tiling);

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_uniform_fused_int64(
    uint32_t blockDim, aclrtStream stream, void* indptr, void* indices,
    void* data, void* rows, void* out_ptr, void* out_rows, void* out_cols,
    void* out_idxs, void* seed_pairs, void* row_split, void* out_starts,
    void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}  // namespace runtime
}  // namespace dgl
#endif  // DGL_USE_ASCEND

#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/aten/csr.h>
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "../array_op.h"
#include "csr_row_wise_sampling_uniform_fused_tiling.h"

namespace dgl {
namespace aten {
namespace impl {

namespace {

// Returns the device's vector-core count, queried at runtime so the
// launch adapts to any SoC (910B family: 40 AIV; other families or
// trimmed vNPU instances differ). Falls back to the arch default when
// the query is unavailable.
uint32_t QueryVectorCoreCountFused(int device_id) {
  int64_t core_num = 0;
  aclError err =
      aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &core_num);
  if (err != ACL_SUCCESS || core_num <= 0 || core_num > 4096) {
    return kDefaultVectorCoreCountFused;
  }
  return static_cast<uint32_t>(core_num);
}

// Returns the per-vector-core unified-buffer budget in bytes (minus the
// runtime-reserved tail). Queried at runtime because UB size differs
// across SoCs (192KB on 910B, 248KB on 950PR).
uint32_t QueryUbAvailableBytesFused(int device_id) {
  int64_t ub_bytes = 0;
  aclError err = aclrtGetDeviceInfo(
      device_id, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, &ub_bytes);
  if (err != ACL_SUCCESS ||
      ub_bytes <= static_cast<int64_t>(kUbReservedBytesFused) ||
      ub_bytes > (1 << 30)) {
    return kDefaultUbBytesFused - kUbReservedBytesFused;
  }
  // Reaching here means ub_bytes > kUbReservedBytesFused (checked above).
  return static_cast<uint32_t>(ub_bytes - kUbReservedBytesFused);
}

// Builds nnz-balanced partitions over row weights (spmm precedent):
// returns num_parts+1 boundaries so each part covers a contiguous row
// range with roughly equal total weight.
std::vector<uint32_t> BuildBalancedPartitionsFused(
    const std::vector<uint32_t>& weights, uint32_t num_parts) {
  std::vector<uint32_t> boundaries(num_parts + 1, 0);
  if (num_parts == 0) return boundaries;

  const uint32_t item_count = static_cast<uint32_t>(weights.size());
  boundaries[num_parts] = item_count;
  if (item_count == 0) return boundaries;

  if (item_count <= num_parts) {
    for (uint32_t i = 0; i <= item_count; ++i) boundaries[i] = i;
    for (uint32_t i = item_count + 1; i <= num_parts; ++i)
      boundaries[i] = item_count;
    return boundaries;
  }

  std::vector<double> prefix(item_count + 1, 0.0);
  for (uint32_t i = 0; i < item_count; ++i)
    prefix[i + 1] = prefix[i] + weights[i];
  const double total_weight = prefix[item_count];
  if (total_weight <= 0.0) {
    for (uint32_t part = 1; part < num_parts; ++part)
      boundaries[part] = part * item_count / num_parts;
    return boundaries;
  }

  for (uint32_t part = 1; part < num_parts; ++part) {
    const double target = total_weight * part / num_parts;
    auto it = std::lower_bound(prefix.begin(), prefix.end(), target);
    boundaries[part] = static_cast<uint32_t>(it - prefix.begin());
  }
  // Enforce monotonicity against ties landing on the same boundary.
  for (uint32_t part = 1; part < num_parts; ++part) {
    if (boundaries[part] < boundaries[part - 1])
      boundaries[part] = boundaries[part - 1];
    if (boundaries[part] > item_count) boundaries[part] = item_count;
  }
  return boundaries;
}

// Uploads a host uint32 table to device memory on the launch stream.
// The stream is synchronized BEFORE the caller's stack buffers go out of
// scope: an async copy only captures the source pointer, so returning
// without a sync would upload stack garbage (spmm precedes its cached
// uploads with the same sync).
void* UploadHostUInt32Fused(
    const std::vector<uint32_t>& host, aclrtStream stream) {
  void* dev = nullptr;
  ASCEND_CALL(aclrtMalloc(
      &dev, host.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpyAsync(
      dev, host.size() * sizeof(uint32_t), host.data(),
      host.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  return dev;
}

}  // namespace

#ifdef DGL_USE_ASCEND
namespace {

// Scatters the compact (rid, pos) pair array into seed_mapping on the
// host (ADR-0014 A1): D2H the pairs, write mapping[rid] = pos, H2D the
// touched mapping back. The mapping is a full device array (length =
// graph nodes), so the round-trip is mapping-sized; acceptable because
// the fused host path already moves mapping-sized arrays per layer.
template <typename IdType>
void ScatterSeedMappingFromPairs(
    IdArray seed_pairs, int64_t pair_count, IdArray seed_mapping,
    int64_t num_rows, aclrtStream stream) {
  if (pair_count <= 0 || !seed_mapping.defined()) return;
  std::vector<IdType> pairs(2 * pair_count);
  ASCEND_CALL(aclrtMemcpy(
      pairs.data(), 2 * pair_count * sizeof(IdType), seed_pairs->data,
      2 * pair_count * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
  const int64_t mapping_len = seed_mapping->shape[0];
  // The mapping spans the graph's node count; the CSR row count is the
  // seed count. A homo graph makes them equal, but heterographs and
  // partial seed sets make mapping_len >= needed rows; both are valid.
  CHECK(mapping_len >= num_rows)
      << "seed_mapping length " << mapping_len
      << " is smaller than the row count " << num_rows;
  std::vector<IdType> mapping(mapping_len);
  ASCEND_CALL(aclrtMemcpy(
      mapping.data(), mapping_len * sizeof(IdType), seed_mapping->data,
      mapping_len * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
  for (int64_t p = 0; p < pair_count; ++p) {
    const IdType rid = pairs[2 * p];
    const IdType pos = pairs[2 * p + 1];
    // Kernel-side validation guarantees rid in [0, num_total_rows);
    // clamp again here (defense in depth, rule: never trust the buffer).
    if (rid >= 0 && rid < static_cast<IdType>(mapping_len)) {
      mapping[rid] = pos;
    }
  }
  ASCEND_CALL(aclrtMemcpyAsync(
      seed_mapping->data, mapping_len * sizeof(IdType), mapping.data(),
      mapping_len * sizeof(IdType), ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ASCEND_CALL(aclrtSynchronizeStream(stream));
}

// Zero-output path: the kernel did not run, but valid rows still need
// their mapping positions (row i of rows maps to seed_mapping[rid]=i).
// rows is a device array; pull it once and scatter directly.
template <typename IdType>
void ScatterSeedMappingFromRows(
    IdArray rows, IdArray seed_mapping, int64_t num_rows,
    aclrtStream stream) {
  if (num_rows <= 0 || !seed_mapping.defined()) return;
  std::vector<IdType> rows_host(num_rows);
  ASCEND_CALL(aclrtMemcpy(
      rows_host.data(), num_rows * sizeof(IdType), rows->data,
      num_rows * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
  const int64_t mapping_len = seed_mapping->shape[0];
  std::vector<IdType> mapping(mapping_len);
  ASCEND_CALL(aclrtMemcpy(
      mapping.data(), mapping_len * sizeof(IdType), seed_mapping->data,
      mapping_len * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
  for (int64_t i = 0; i < num_rows; ++i) {
    const IdType rid = rows_host[i];
    if (rid >= 0 && rid < static_cast<IdType>(mapping_len)) {
      mapping[rid] = static_cast<IdType>(i);
    }
  }
  ASCEND_CALL(aclrtMemcpyAsync(
      seed_mapping->data, mapping_len * sizeof(IdType), mapping.data(),
      mapping_len * sizeof(IdType), ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ASCEND_CALL(aclrtSynchronizeStream(stream));
}

}  // namespace
#endif  // DGL_USE_ASCEND

template <DGLDeviceType XPU, typename IdType, bool map_seed_nodes>
std::pair<CSRMatrix, IdArray> CSRRowWiseSamplingUniformFused(
    CSRMatrix mat, IdArray rows, IdArray seed_mapping,
    std::vector<IdType>* new_seed_nodes, int64_t num_samples, bool replace) {
#ifdef DGL_USE_ASCEND
  auto ctx = mat.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend)
      << "Expected Ascend device context for "
         "CSRRowWiseSamplingUniformFused";
  // Defense in depth (ADR-0012): the upstream Python layer historically
  // created the mapping as int64 regardless of graph idtype, which the
  // CPU path silently misreads (4-byte offset UB). Reject it here.
  if (map_seed_nodes) {
    CHECK(seed_mapping.defined() &&
          seed_mapping->dtype == mat.indptr->dtype)
        << "seed_mapping dtype must match the CSR idtype "
           "(got bits="
        << (seed_mapping.defined() ? seed_mapping->dtype.bits : 0)
        << ", want " << mat.indptr->dtype.bits << ")";
  }
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  const bool select_all = (num_samples == -1);
  replace = (replace && !select_all);

  const int64_t num_rows = rows->shape[0];
  const uint8_t nbits = mat.indptr->dtype.bits;

  if (map_seed_nodes && new_seed_nodes != nullptr) {
    new_seed_nodes->clear();
  }

  // Early exit: nothing to sample. The block CSR still needs its indptr
  // (num_rows + 1 zeros) so downstream consumers see a well-formed empty
  // matrix (CPU reference allocates block_csr_indptr unconditionally).
  if (num_rows == 0 || mat.indptr->shape[0] <= 1 || num_samples == 0) {
    IdArray block_csr_indptr = aten::NewIdArray(num_rows + 1, ctx, nbits);
    IdArray empty = aten::NewIdArray(0, ctx, nbits);
    // Zero indptr: DGL's allocator reuses memory without zeroing.
    const int64_t indptr_bytes = (num_rows + 1) * (nbits / 8);
    auto stream0 = dgl::runtime::getCurrentAscendStream();
    if (num_rows + 1 > 0) {
      ASCEND_CALL(aclrtMemsetAsync(
          block_csr_indptr->data, indptr_bytes, 0, indptr_bytes, stream0));
      ASCEND_CALL(aclrtSynchronizeStream(stream0));
    }
    if (map_seed_nodes && new_seed_nodes != nullptr && num_rows > 0) {
      // CPU reference copies rows into new_seed_nodes even on early exit
      // only when sampling ran; on the empty path it stays empty. But the
      // seed mapping positions for valid rows must still be written when
      // the kernel would have done so — for the degenerate cases here the
      // CPU path also leaves the mapping untouched (no rows processed).
      new_seed_nodes->clear();
    }
    return std::make_pair(
        CSRMatrix(num_rows, 0, block_csr_indptr, empty, empty), empty);
  }

  // Per-row pick counts (degrees come back to the host once).
  const uint32_t fanout = select_all ? 0u : static_cast<uint32_t>(num_samples);
  NDArray deg = CSRGetRowNNZ<kDGLAscend, IdType>(mat, rows);
  std::vector<IdType> deg_host(num_rows);
  ASCEND_CALL(aclrtMemcpy(
      deg_host.data(), num_rows * sizeof(IdType), deg->data,
      num_rows * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
  std::vector<uint32_t> picks(num_rows);
  for (int64_t i = 0; i < num_rows; ++i) {
    const uint32_t d = static_cast<uint32_t>(deg_host[i]);
    picks[i] = select_all ? d
               : replace  ? (d == 0 ? 0u : fanout)
                          : std::min(fanout, d);
  }

  // nnz-balanced row partitions across all vector cores (spmm pattern).
  const uint32_t block_dim = QueryVectorCoreCountFused(ctx.device_id);
  const std::vector<uint32_t> row_split =
      BuildBalancedPartitionsFused(picks, block_dim);

  // Per-block output offsets as prefix sums of picks over row ranges.
  std::vector<uint32_t> out_starts(block_dim + 1, 0);
  {
    std::vector<uint32_t> prefix(num_rows + 1, 0);
    for (int64_t i = 0; i < num_rows; ++i) prefix[i + 1] = prefix[i] + picks[i];
    for (uint32_t b = 0; b <= block_dim; ++b)
      out_starts[b] = prefix[row_split[b]];
  }
  const int64_t max_output = out_starts[block_dim];
  CHECK(max_output <= static_cast<int64_t>(std::numeric_limits<IdType>::max()))
      << "Output size " << max_output << " exceeds IdType range";

  auto stream = dgl::runtime::getCurrentAscendStream();
  const bool has_data = aten::CSRHasData(mat);
  void* data_ptr = has_data ? mat.data->data : nullptr;

  uint32_t tiling_data[kTilingHeaderWordsFused] = {
      static_cast<uint32_t>(num_rows),
      fanout,
      static_cast<uint32_t>(replace ? 1 : 0),
      static_cast<uint32_t>(has_data ? 1 : 0),
      static_cast<uint32_t>(RandomEngine::ThreadLocal()->RandInt(1000000000)),
      static_cast<uint32_t>(select_all ? 1 : 0),
      static_cast<uint32_t>(mat.num_rows),
      QueryUbAvailableBytesFused(ctx.device_id),
      static_cast<uint32_t>(map_seed_nodes ? 1 : 0),
  };

  IdArray picked_coo_rows = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_col = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_idx = aten::NewIdArray(max_output, ctx, nbits);
  IdArray block_csr_indptr = aten::NewIdArray(num_rows + 1, ctx, nbits);
  // Compact (rid, pos) pair array: one pair per valid row when mapping.
  // When map_seed_nodes is false the kernel never touches it; a 1-element
  // dummy keeps the GlobalTensor binding safe (ADR-0014).
  const int64_t pair_count = map_seed_nodes ? num_rows : 1;
  IdArray seed_pairs = aten::NewIdArray(pair_count, ctx, nbits);

  if (max_output == 0) {
    // No edges: indptr must still be a valid all-zero prefix (the kernel
    // is not launched; zero it explicitly — allocator reuse).
    const int64_t indptr_bytes = (num_rows + 1) * (nbits / 8);
    ASCEND_CALL(aclrtMemsetAsync(
        block_csr_indptr->data, indptr_bytes, 0, indptr_bytes, stream));
    // Seed pairs are still produced for valid rows (pos writes) — but
    // with zero output the kernel did not run, so scatter nothing. The
    // CPU reference in this regime processes rows and writes mapping
    // entries; replicate that by scattering on the host directly from
    // rows (all rows map to their positions; degenerate picks are zero
    // but the mapping is row-position based, independent of picks).
    if (map_seed_nodes) {
      ScatterSeedMappingFromRows<IdType>(
          rows, seed_mapping, num_rows, stream);
      if (new_seed_nodes != nullptr) {
        new_seed_nodes->resize(num_rows);
        ASCEND_CALL(aclrtMemcpy(
            new_seed_nodes->data(), num_rows * sizeof(IdType), rows->data,
            num_rows * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
      }
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));
    return std::make_pair(
        CSRMatrix(num_rows, 0, block_csr_indptr, picked_col.CreateView({0}, picked_col->dtype),
                  picked_idx.CreateView({0}, picked_idx->dtype)),
        picked_coo_rows.CreateView({0}, picked_coo_rows->dtype));
  }

  // Zero the output buffers on the launch stream (spmm pattern): DGL's
  // array allocator reuses device memory without zeroing, so slots the
  // kernel does not write (idle blocks) must read as 0, not stale data
  // from earlier launches.
  const int64_t out_bytes = max_output * (nbits / 8);
  ASCEND_CALL(
      aclrtMemsetAsync(picked_coo_rows->data, out_bytes, 0, out_bytes, stream));
  ASCEND_CALL(
      aclrtMemsetAsync(picked_col->data, out_bytes, 0, out_bytes, stream));
  ASCEND_CALL(
      aclrtMemsetAsync(picked_idx->data, out_bytes, 0, out_bytes, stream));
  const int64_t indptr_bytes = (num_rows + 1) * (nbits / 8);
  ASCEND_CALL(aclrtMemsetAsync(
      block_csr_indptr->data, indptr_bytes, 0, indptr_bytes, stream));
  const int64_t pairs_bytes = pair_count * (nbits / 8);
  ASCEND_CALL(
      aclrtMemsetAsync(seed_pairs->data, pairs_bytes, 0, pairs_bytes, stream));

  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(
      &tiling_dev, sizeof(tiling_data), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpyAsync(
      tiling_dev, sizeof(tiling_data), tiling_data, sizeof(tiling_data),
      ACL_MEMCPY_HOST_TO_DEVICE, stream));
  // tiling_data is a stack array: wait for the copy to land before the
  // frame that owns it returns.
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  void* row_split_dev = UploadHostUInt32Fused(row_split, stream);
  void* out_starts_dev = UploadHostUInt32Fused(out_starts, stream);

  if (std::is_same<IdType, int32_t>::value) {
    aclError err = aclrtlaunch_csr_row_wise_sampling_uniform_fused_int32(
        block_dim, stream, mat.indptr->data, mat.indices->data, data_ptr,
        rows->data, block_csr_indptr->data, picked_coo_rows->data,
        picked_col->data, picked_idx->data, seed_pairs->data, row_split_dev,
        out_starts_dev, tiling_dev);
    CHECK(err == ACL_SUCCESS)
        << "csr_row_wise_sampling_uniform_fused_int32 launch failed: "
        << err;
  } else {
    aclError err = aclrtlaunch_csr_row_wise_sampling_uniform_fused_int64(
        block_dim, stream, mat.indptr->data, mat.indices->data, data_ptr,
        rows->data, block_csr_indptr->data, picked_coo_rows->data,
        picked_col->data, picked_idx->data, seed_pairs->data, row_split_dev,
        out_starts_dev, tiling_dev);
    CHECK(err == ACL_SUCCESS)
        << "csr_row_wise_sampling_uniform_fused_int64 launch failed: "
        << err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tiling_dev));
  ASCEND_CALL(aclrtFree(row_split_dev));
  ASCEND_CALL(aclrtFree(out_starts_dev));

  // ADR-0014 A1 closure: scatter the compact (rid, pos) pairs into
  // seed_mapping on the host, then write the mapping back. The kernel is
  // already synchronized, so a plain D2H copy is safe.
  if (map_seed_nodes) {
    ScatterSeedMappingFromPairs<IdType>(seed_pairs, num_rows, seed_mapping,
                                        num_rows, stream);
    if (new_seed_nodes != nullptr) {
      new_seed_nodes->resize(num_rows);
      ASCEND_CALL(aclrtMemcpy(
          new_seed_nodes->data(), num_rows * sizeof(IdType), rows->data,
          num_rows * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
    }
  }

  return std::make_pair(
      CSRMatrix(num_rows, max_output, block_csr_indptr, picked_col,
                picked_idx),
      picked_coo_rows);
#else
  LOG(FATAL) << "Ascend support is not compiled. "
                "Please compile with -DUSE_ASCEND=ON";
  return {};
#endif  // DGL_USE_ASCEND
}

template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingUniformFused<kDGLAscend, int32_t, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingUniformFused<kDGLAscend, int64_t, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingUniformFused<kDGLAscend, int32_t, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingUniformFused<kDGLAscend, int64_t, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
