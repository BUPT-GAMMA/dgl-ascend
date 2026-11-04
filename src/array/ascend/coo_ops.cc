/**
 * @file src/array/ascend/coo_ops.cc
 * @brief Ascend NPU implementations of COO operators.
 *
 * Most operators compose native CSR primitives. COOGetRowDataAndIndices uses
 * a direct Count / host-prefix / Scatter implementation to preserve input COO
 * order and implicit edge IDs.
 */
#include <acl/acl_rt.h>

#include <dgl/array.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "../array_op.h"
#include "coo_get_row_data_and_indices_tiling.h"

#define ASCEND_CALL(func)                                    \
  {                                                          \
    aclError e = (func);                                     \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}  // namespace runtime
}  // namespace dgl

extern "C" uint32_t aclrtlaunch_coo_get_row_count_i32(
    uint32_t blockDim, aclrtStream stream, void* rows, void* count_slabs,
    void* mask_workspace, void* tiling);
extern "C" uint32_t aclrtlaunch_coo_get_row_scatter_i32(
    uint32_t blockDim, aclrtStream stream, void* cols, void* data,
    void* offsets, void* mask_workspace, void* out_data, void* out_cols,
    void* tiling);
extern "C" uint32_t aclrtlaunch_coo_get_row_count_i64(
    uint32_t blockDim, aclrtStream stream, void* rows, void* count_slabs,
    void* mask_workspace, void* tiling);
extern "C" uint32_t aclrtlaunch_coo_get_row_scatter_i64(
    uint32_t blockDim, aclrtStream stream, void* cols, void* data,
    void* offsets, void* mask_workspace, void* out_data, void* out_cols,
    void* tiling);

namespace dgl {
namespace aten {
namespace impl {

#ifdef DGL_USE_ASCEND

namespace {

using coo_get_row_v2::GetRowTiling;
using coo_get_row_v2::kChunkElements;
using coo_get_row_v2::kCountStrideWords;
using coo_get_row_v2::kMaskWorkspaceBytesPerChunk;
using coo_get_row_v2::kMaxBlocks;

class GetRowWorkspaces {
 public:
  void Allocate(
      uint32_t block_dim, uint32_t mask_chunks_per_block,
      size_t tiling_bytes) {
    ASCEND_CALL(aclrtMalloc(
        reinterpret_cast<void**>(&count_slabs),
        static_cast<size_t>(block_dim) * kCountStrideWords * sizeof(uint32_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CALL(aclrtMalloc(
        &mask_workspace,
        static_cast<size_t>(block_dim) * mask_chunks_per_block *
            kMaskWorkspaceBytesPerChunk,
        ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CALL(aclrtMalloc(
        reinterpret_cast<void**>(&offsets),
        static_cast<size_t>(block_dim + 1) * sizeof(uint32_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CALL(aclrtMalloc(
        &tiling, tiling_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  }

  void Free() {
    FreeOne(reinterpret_cast<void**>(&count_slabs));
    FreeOne(&mask_workspace);
    FreeOne(reinterpret_cast<void**>(&offsets));
    FreeOne(&tiling);
  }

  ~GetRowWorkspaces() {
    FreeUnchecked(count_slabs);
    FreeUnchecked(mask_workspace);
    FreeUnchecked(offsets);
    FreeUnchecked(tiling);
  }

  GetRowWorkspaces() = default;
  GetRowWorkspaces(const GetRowWorkspaces&) = delete;
  GetRowWorkspaces& operator=(const GetRowWorkspaces&) = delete;

  uint32_t* count_slabs = nullptr;
  void* mask_workspace = nullptr;
  uint32_t* offsets = nullptr;
  void* tiling = nullptr;

 private:
  static void FreeOne(void** ptr) {
    if (*ptr == nullptr) return;
    void* allocated = *ptr;
    *ptr = nullptr;
    ASCEND_CALL(aclrtFree(allocated));
  }

  template <typename T>
  static void FreeUnchecked(T*& ptr) {
    if (ptr != nullptr) {
      aclrtFree(ptr);
      ptr = nullptr;
    }
  }
};

template <typename IdType>
void CheckGetRowArray(
    const NDArray& array, const DGLContext& ctx, const char* name) {
  // defined() 判断 NDArray 是否实际存在。
  // 不使用 IsNullArray()，因为 IsNullArray() 会把合法的零长度数组
  // 也判断为 null。
  CHECK(array.defined()) << name << " must be defined";

  CHECK_EQ(array->ndim, 1) << name << " must be one-dimensional";
  CHECK_EQ(array->dtype.code, kDGLInt)
      << name << " must contain signed integer indices";
  CHECK_EQ(array->dtype.bits, sizeof(IdType) * 8)
      << name << " has an unexpected integer width";
  CHECK_EQ(array->dtype.lanes, 1) << name << " must have one dtype lane";
  CHECK_EQ(array->ctx.device_type, ctx.device_type)
      << name << " must be on the COO device";
  CHECK_EQ(array->ctx.device_id, ctx.device_id)
      << name << " must be on the COO device";
}

uint32_t MaskChunksPerBlock(uint32_t nnz, uint32_t block_dim) {
  const uint32_t max_block_elements =
      nnz / block_dim + (nnz % block_dim == 0 ? 0U : 1U);
  return max_block_elements / kChunkElements +
         (max_block_elements % kChunkElements == 0 ? 0U : 1U);
}

template <typename IdType>
uint32_t LaunchGetRowCount(
    uint32_t block_dim, aclrtStream stream, void* rows, void* count_slabs,
    void* mask_workspace, void* tiling) {
  if constexpr (std::is_same<IdType, int32_t>::value) {
    return ACLRT_LAUNCH_KERNEL(coo_get_row_count_i32)(
        block_dim, stream, rows, count_slabs, mask_workspace, tiling);
  } else {
    return ACLRT_LAUNCH_KERNEL(coo_get_row_count_i64)(
        block_dim, stream, rows, count_slabs, mask_workspace, tiling);
  }
}

template <typename IdType>
uint32_t LaunchGetRowScatter(
    uint32_t block_dim, aclrtStream stream, void* cols, void* data,
    void* offsets, void* mask_workspace, void* out_data, void* out_cols,
    void* tiling) {
  if constexpr (std::is_same<IdType, int32_t>::value) {
    return ACLRT_LAUNCH_KERNEL(coo_get_row_scatter_i32)(
        block_dim, stream, cols, data, offsets, mask_workspace, out_data,
        out_cols, tiling);
  } else {
    return ACLRT_LAUNCH_KERNEL(coo_get_row_scatter_i64)(
        block_dim, stream, cols, data, offsets, mask_workspace, out_data,
        out_cols, tiling);
  }
}

// Ensure the COO matrix carries an explicit data array. When the COO has no
// explicit data, Ascend COOToCSR (coo2csr.cc) renumbers the CSR data array in
// CSR-sorted order instead of carrying the original COO edge positions. This
// breaks eid semantics for downstream CSR-composed ops (COOSliceRows /
// COOSliceMatrix) whose results depend on eid == original COO position.  The
// native COOGetRowDataAndIndices V2 path below does not use this helper.
//
// Mirrors the CPU COOSort_/COOToCSR convention (coo_sort.cc:157-158,
// coo2csr.cc:184-188): when there is no data, materialize Range(0, nnz) as the
// data array *before* calling COOToCSR. With explicit data present,
// COOToCSR gathers it by the sort permutation, preserving original positions.
template <typename IdType>
void EnsureCOODataForCSR(COOMatrix* coo) {
  if (COOHasData(*coo)) {
    return;
  }
  const int64_t nnz = coo->row->shape[0];
  // Generate Range(0, nnz) on CPU then copy to NPU (aten::Range has no Ascend
  // kernel; this matches the pattern in coo2csr.cc:173-175 / :186-188).
  aclrtSynchronizeDevice();
  DGLContext cpu_ctx{kDGLCPU, 0};
  IdArray cpu_data = aten::Range(0, nnz, coo->row->dtype.bits, cpu_ctx);
  coo->data = cpu_data.CopyTo(coo->row->ctx);
}

}  // anonymous namespace

// =========================================================================
// COOSort_
// =========================================================================

template <typename IdType>
void COOSort_Ascend(COOMatrix* coo, bool sort_column) {
  if (sort_column) {
    // Full sort (row + column): fall back to CPU since CSRSort_ is not
    // yet Ascend-native. The COOToCSR step already sorts by row, but
    // column sorting within each row requires CSRSort_ which currently
    // has no Ascend kernel.
    DGLContext cpu_ctx{kDGLCPU, 0};
    aclrtSynchronizeDevice();
    COOMatrix coo_cpu{
        coo->num_rows, coo->num_cols,
        coo->row.CopyTo(cpu_ctx), coo->col.CopyTo(cpu_ctx),
        coo->data.CopyTo(cpu_ctx), coo->row_sorted, coo->col_sorted};
    COOSort_<kDGLCPU, IdType>(&coo_cpu, sort_column);
    coo->row = coo_cpu.row.CopyTo(coo->row->ctx);
    coo->col = coo_cpu.col.CopyTo(coo->row->ctx);
    coo->data = coo_cpu.data.CopyTo(coo->row->ctx);
    coo->row_sorted = coo_cpu.row_sorted;
    coo->col_sorted = coo_cpu.col_sorted;
    return;
  }

  // Row-only sort: COOToCSR sorts by row, CSRToCOO gives row-sorted COO.
  // Ensure the COO carries explicit data so that COOToCSR preserves the
  // original edge positions (Range(0, nnz)) through the sort permutation.
  EnsureCOODataForCSR<IdType>(coo);
  CSRMatrix csr = COOToCSR(*coo);
  COOMatrix sorted_coo = CSRToCOO(csr, false);
  coo->row = sorted_coo.row;
  coo->col = sorted_coo.col;
  coo->data = sorted_coo.data;
  coo->row_sorted = true;
  // col_sorted: preserved if rows were already sorted; otherwise unknown.
  coo->col_sorted = coo->col_sorted && coo->row_sorted;
}

template <>
void COOSort_<kDGLAscend, int32_t>(COOMatrix* coo, bool sort_column) {
  COOSort_Ascend<int32_t>(coo, sort_column);
}

template <>
void COOSort_<kDGLAscend, int64_t>(COOMatrix* coo, bool sort_column) {
  COOSort_Ascend<int64_t>(coo, sort_column);
}

// =========================================================================
// COOGetRowNNZ (scalar)
// =========================================================================

template <typename IdType>
int64_t COOGetRowNNZ_Ascend(COOMatrix coo, int64_t row) {
  CSRMatrix csr = COOToCSR(coo);
  return CSRGetRowNNZ<kDGLAscend, IdType>(csr, row);
}

template <>
int64_t COOGetRowNNZ<kDGLAscend, int32_t>(COOMatrix coo, int64_t row) {
  return COOGetRowNNZ_Ascend<int32_t>(coo, row);
}

template <>
int64_t COOGetRowNNZ<kDGLAscend, int64_t>(COOMatrix coo, int64_t row) {
  return COOGetRowNNZ_Ascend<int64_t>(coo, row);
}

// =========================================================================
// COOGetRowNNZ (array)
// =========================================================================

template <typename IdType>
NDArray COOGetRowNNZ_Ascend(COOMatrix coo, NDArray rows) {
  CSRMatrix csr = COOToCSR(coo);
  return CSRGetRowNNZ<kDGLAscend, IdType>(csr, rows);
}

template <>
NDArray COOGetRowNNZ<kDGLAscend, int32_t>(COOMatrix coo, NDArray rows) {
  return COOGetRowNNZ_Ascend<int32_t>(coo, rows);
}

template <>
NDArray COOGetRowNNZ<kDGLAscend, int64_t>(COOMatrix coo, NDArray rows) {
  return COOGetRowNNZ_Ascend<int64_t>(coo, rows);
}

// =========================================================================
// COOGetRowDataAndIndices
// =========================================================================

template <typename IdType>
std::pair<NDArray, NDArray> COOGetRowDataAndIndices_Ascend(
    COOMatrix coo, int64_t row) {
  CHECK_GE(coo.num_rows, 0) << "COO num_rows must be non-negative";
  CHECK(row >= 0 && row < coo.num_rows) << "Invalid row index: " << row;
  CHECK(coo.row.defined()) << "COO row must be defined";

  const DGLContext ctx = coo.row->ctx;
  CHECK_EQ(ctx.device_type, kDGLAscend) << "Expected Ascend device context";
  CheckGetRowArray<IdType>(coo.row, ctx, "COO row");
  CheckGetRowArray<IdType>(coo.col, ctx, "COO col");

  const int64_t nnz64 = coo.row->shape[0];
  CHECK_GE(nnz64, 0) << "COO nnz must be non-negative";
  CHECK_EQ(coo.col->shape[0], nnz64)
      << "COO row and col lengths must match";
  const bool has_data = COOHasData(coo);
  if (has_data) {
    CheckGetRowArray<IdType>(coo.data, ctx, "COO data");
    CHECK_EQ(coo.data->shape[0], nnz64)
        << "COO data length must equal nnz";
  }

  if constexpr (std::is_same<IdType, int32_t>::value) {
    CHECK_LE(coo.num_rows, std::numeric_limits<int32_t>::max())
        << "COO num_rows does not fit int32";
    CHECK_LE(nnz64, std::numeric_limits<int32_t>::max())
        << "COO nnz exceeds the int32 implicit-EID range";
  } else {
    CHECK_LE(
        static_cast<uint64_t>(nnz64),
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
        << "COO nnz exceeds the uint32 control range";
  }

  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  if (nnz64 == 0) {
    NDArray empty_data = NDArray::Empty({0}, coo.row->dtype, ctx);
    NDArray empty_cols = NDArray::Empty({0}, coo.col->dtype, ctx);
    return std::make_pair(empty_data, empty_cols);
  }

  const uint32_t nnz = static_cast<uint32_t>(nnz64);
  const uint32_t block_dim = std::min<uint32_t>(kMaxBlocks, nnz);
  const uint32_t mask_chunks_per_block =
      MaskChunksPerBlock(nnz, block_dim);

  GetRowTiling<IdType> tiling_host{};
  tiling_host.nnz = nnz;
  tiling_host.target_row = static_cast<IdType>(row);
  tiling_host.has_data = has_data ? 1U : 0U;
  tiling_host.mask_chunks_per_block = mask_chunks_per_block;

  GetRowWorkspaces ws;
  ws.Allocate(block_dim, mask_chunks_per_block, sizeof(tiling_host));
  ASCEND_CALL(aclrtMemcpy(
      ws.tiling, sizeof(tiling_host), &tiling_host, sizeof(tiling_host),
      ACL_MEMCPY_HOST_TO_DEVICE));

  // Both stages use DGL's current stream.  The only inter-stage sync is
  // required to bring the per-block counts to the host for the prefix sum.
  aclrtStream stream = dgl::runtime::getCurrentAscendStream();
  const uint32_t count_launch = LaunchGetRowCount<IdType>(
      block_dim, stream, coo.row->data, ws.count_slabs, ws.mask_workspace,
      ws.tiling);
  CHECK_EQ(count_launch, ACL_SUCCESS)
      << "COOGetRowDataAndIndices count kernel launch failed: "
      << count_launch;
  ASCEND_CALL(aclrtSynchronizeStream(stream));

  const size_t count_words =
      static_cast<size_t>(block_dim) * kCountStrideWords;
  std::vector<uint32_t> count_host(count_words, 0);
  ASCEND_CALL(aclrtMemcpy(
      count_host.data(), count_words * sizeof(uint32_t), ws.count_slabs,
      count_words * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST));

  std::vector<uint32_t> offsets_host(block_dim + 1, 0);
  uint64_t prefix = 0;
  for (uint32_t block = 0; block < block_dim; ++block) {
    prefix += count_host[block * kCountStrideWords];
    CHECK_LE(prefix, static_cast<uint64_t>(nnz))
        << "COOGetRowDataAndIndices count exceeds input nnz";
    offsets_host[block + 1] = static_cast<uint32_t>(prefix);
  }
  const uint32_t output_size = offsets_host.back();

  NDArray out_data =
      NDArray::Empty({static_cast<int64_t>(output_size)}, coo.row->dtype, ctx);
  NDArray out_cols =
      NDArray::Empty({static_cast<int64_t>(output_size)}, coo.col->dtype, ctx);
  if (output_size == 0) {
    ws.Free();
    return std::make_pair(out_data, out_cols);
  }

  ASCEND_CALL(aclrtMemcpy(
      ws.offsets, offsets_host.size() * sizeof(uint32_t),
      offsets_host.data(), offsets_host.size() * sizeof(uint32_t),
      ACL_MEMCPY_HOST_TO_DEVICE));

  const uint32_t scatter_launch = LaunchGetRowScatter<IdType>(
      block_dim, stream, coo.col->data,
      has_data ? coo.data->data : nullptr, ws.offsets, ws.mask_workspace,
      out_data->data, out_cols->data, ws.tiling);
  CHECK_EQ(scatter_launch, ACL_SUCCESS)
      << "COOGetRowDataAndIndices scatter kernel launch failed: "
      << scatter_launch;

  // Raw ACL workspace and tiling must outlive Scatter; without a stream-ordered
  // free, this completion sync is required before releasing them.
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ws.Free();
  return std::make_pair(out_data, out_cols);
}

template <>
std::pair<NDArray, NDArray> COOGetRowDataAndIndices<kDGLAscend, int32_t>(
    COOMatrix coo, int64_t row) {
  return COOGetRowDataAndIndices_Ascend<int32_t>(coo, row);
}

template <>
std::pair<NDArray, NDArray> COOGetRowDataAndIndices<kDGLAscend, int64_t>(
    COOMatrix coo, int64_t row) {
  return COOGetRowDataAndIndices_Ascend<int64_t>(coo, row);
}

// =========================================================================
// COOSliceRows (start, end)
// =========================================================================

template <typename IdType>
COOMatrix COOSliceRows_Ascend(COOMatrix coo, int64_t start, int64_t end) {
  // Ensure explicit data so eid reflects original COO positions.
  EnsureCOODataForCSR<IdType>(&coo);
  CSRMatrix csr = COOToCSR(coo);
  CSRMatrix sliced = CSRSliceRows<kDGLAscend, IdType>(csr, start, end);
  return CSRToCOO(sliced, false);
}

template <>
COOMatrix COOSliceRows<kDGLAscend, int32_t>(COOMatrix coo, int64_t start, int64_t end) {
  return COOSliceRows_Ascend<int32_t>(coo, start, end);
}

template <>
COOMatrix COOSliceRows<kDGLAscend, int64_t>(COOMatrix coo, int64_t start, int64_t end) {
  return COOSliceRows_Ascend<int64_t>(coo, start, end);
}

// =========================================================================
// COOSliceRows (rows)
// =========================================================================

template <typename IdType>
COOMatrix COOSliceRows_Ascend(COOMatrix coo, NDArray rows) {
  // Ensure explicit data so eid reflects original COO positions.
  EnsureCOODataForCSR<IdType>(&coo);
  CSRMatrix csr = COOToCSR(coo);
  CSRMatrix sliced = CSRSliceRows<kDGLAscend, IdType>(csr, rows);
  return CSRToCOO(sliced, false);
}

template <>
COOMatrix COOSliceRows<kDGLAscend, int32_t>(COOMatrix coo, NDArray rows) {
  return COOSliceRows_Ascend<int32_t>(coo, rows);
}

template <>
COOMatrix COOSliceRows<kDGLAscend, int64_t>(COOMatrix coo, NDArray rows) {
  return COOSliceRows_Ascend<int64_t>(coo, rows);
}

// =========================================================================
// COOSliceMatrix
// =========================================================================

template <typename IdType>
COOMatrix COOSliceMatrix_Ascend(COOMatrix coo, NDArray rows, NDArray cols) {
  // Ensure explicit data so eid reflects original COO positions.
  EnsureCOODataForCSR<IdType>(&coo);
  CSRMatrix csr = COOToCSR(coo);
  CSRMatrix sliced = CSRSliceMatrix<kDGLAscend, IdType>(csr, rows, cols);
  return CSRToCOO(sliced, false);
}

template <>
COOMatrix COOSliceMatrix<kDGLAscend, int32_t>(COOMatrix coo, NDArray rows, NDArray cols) {
  return COOSliceMatrix_Ascend<int32_t>(coo, rows, cols);
}

template <>
COOMatrix COOSliceMatrix<kDGLAscend, int64_t>(COOMatrix coo, NDArray rows, NDArray cols) {
  return COOSliceMatrix_Ascend<int64_t>(coo, rows, cols);
}

#endif  // DGL_USE_ASCEND

}  // namespace impl
}  // namespace aten
}  // namespace dgl
