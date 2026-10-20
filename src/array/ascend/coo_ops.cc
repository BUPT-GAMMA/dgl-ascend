/**
 * @file src/array/ascend/coo_ops.cc
 * @brief Ascend NPU implementations of COO operators.
 *
 * Most operators use COO → CSR composition. COOGetRowDataAndIndices uses a
 * dedicated stable Count/Scatter implementation to preserve original COO EIDs.
 */
#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <dgl/array.h>
#include <limits>
#include <type_traits>
#include <vector>

#include "../array_op.h"
#include "coo_get_row_data_and_indices_tiling.h"

#ifdef DGL_USE_ASCEND
extern "C" uint32_t aclrtlaunch_coo_get_row_count_i32(
    uint32_t blockDim, aclrtStream stream, void* rows, void* count_slabs,
    void* tiling);

extern "C" uint32_t aclrtlaunch_coo_get_row_count_i64(
    uint32_t blockDim, aclrtStream stream, void* rows, void* count_slabs,
    void* tiling);

extern "C" uint32_t aclrtlaunch_coo_get_row_scatter_i32(
    uint32_t blockDim, aclrtStream stream, void* rows, void* cols, void* data,
    void* offsets, void* out_data, void* out_cols, void* tiling);

extern "C" uint32_t aclrtlaunch_coo_get_row_scatter_i64(
    uint32_t blockDim, aclrtStream stream, void* rows, void* cols, void* data,
    void* offsets, void* out_data, void* out_cols, void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}  // namespace runtime
}  // namespace dgl
#endif

namespace dgl {
namespace aten {
namespace impl {

#ifdef DGL_USE_ASCEND

namespace {

void CheckCooGetRowAcl(aclError error, const char* operation) {
  CHECK(error == ACL_SUCCESS)
      << operation << " failed with Ascend error code " << error;
}

struct CooGetRowWorkspaces {
  uint32_t* count_slabs = nullptr;
  uint32_t* offsets = nullptr;
  void* tiling = nullptr;

  void Allocate(uint32_t block_dim, size_t tiling_bytes) {
    const size_t count_bytes =
        static_cast<size_t>(block_dim) * kCooGetRowCountStrideWords *
        sizeof(uint32_t);
    const size_t offset_bytes =
        (static_cast<size_t>(block_dim) + 1) * sizeof(uint32_t);
    CheckCooGetRowAcl(
        aclrtMalloc(
            reinterpret_cast<void**>(&count_slabs), count_bytes,
            ACL_MEM_MALLOC_HUGE_FIRST),
        "COOGetRowDataAndIndices count workspace allocation");
    CheckCooGetRowAcl(
        aclrtMalloc(
            reinterpret_cast<void**>(&offsets), offset_bytes,
            ACL_MEM_MALLOC_HUGE_FIRST),
        "COOGetRowDataAndIndices offsets workspace allocation");
    CheckCooGetRowAcl(
        aclrtMalloc(&tiling, tiling_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
        "COOGetRowDataAndIndices tiling allocation");
  }

  void Free() {
    if (count_slabs != nullptr) {
      uint32_t* ptr = count_slabs;
      count_slabs = nullptr;
      CheckCooGetRowAcl(
          aclrtFree(ptr),
          "COOGetRowDataAndIndices count workspace free");
    }
    if (offsets != nullptr) {
      uint32_t* ptr = offsets;
      offsets = nullptr;
      CheckCooGetRowAcl(
          aclrtFree(ptr),
          "COOGetRowDataAndIndices offsets workspace free");
    }
    if (tiling != nullptr) {
      void* ptr = tiling;
      tiling = nullptr;
      CheckCooGetRowAcl(
          aclrtFree(ptr), "COOGetRowDataAndIndices tiling free");
    }
  }

  ~CooGetRowWorkspaces() {
    // Best-effort cleanup only: CHECK throws in this repository's default
    // logging configuration and must not escape a destructor during unwinding.
    if (count_slabs != nullptr) {
      (void)aclrtFree(count_slabs);
    }
    if (offsets != nullptr) {
      (void)aclrtFree(offsets);
    }
    if (tiling != nullptr) {
      (void)aclrtFree(tiling);
    }
  }
  CooGetRowWorkspaces() = default;
  CooGetRowWorkspaces(const CooGetRowWorkspaces&) = delete;
  CooGetRowWorkspaces& operator=(const CooGetRowWorkspaces&) = delete;
};

template <typename IdType>
void LaunchCooGetRowCount(
    uint32_t block_dim, aclrtStream stream, const void* rows,
    uint32_t* count_slabs, void* tiling) {
  static_assert(
      std::is_same<IdType, int32_t>::value ||
          std::is_same<IdType, int64_t>::value,
      "COOGetRowDataAndIndices only supports int32 and int64 IDs");
  uint32_t error = ACL_SUCCESS;
  if (std::is_same<IdType, int32_t>::value) {
    error = aclrtlaunch_coo_get_row_count_i32(
        block_dim, stream, const_cast<void*>(rows), count_slabs, tiling);
  } else {
    error = aclrtlaunch_coo_get_row_count_i64(
        block_dim, stream, const_cast<void*>(rows), count_slabs, tiling);
  }
  CHECK(error == ACL_SUCCESS)
      << "COOGetRowDataAndIndices count launch failed: " << error;
}

template <typename IdType>
void LaunchCooGetRowScatter(
    uint32_t block_dim, aclrtStream stream, const void* rows,
    const void* cols, const void* data, uint32_t* offsets, void* out_data,
    void* out_cols, void* tiling) {
  static_assert(
      std::is_same<IdType, int32_t>::value ||
          std::is_same<IdType, int64_t>::value,
      "COOGetRowDataAndIndices only supports int32 and int64 IDs");
  uint32_t error = ACL_SUCCESS;
  if (std::is_same<IdType, int32_t>::value) {
    error = aclrtlaunch_coo_get_row_scatter_i32(
        block_dim, stream, const_cast<void*>(rows), const_cast<void*>(cols),
        const_cast<void*>(data), offsets, out_data, out_cols, tiling);
  } else {
    error = aclrtlaunch_coo_get_row_scatter_i64(
        block_dim, stream, const_cast<void*>(rows), const_cast<void*>(cols),
        const_cast<void*>(data), offsets, out_data, out_cols, tiling);
  }
  CHECK(error == ACL_SUCCESS)
      << "COOGetRowDataAndIndices scatter launch failed: " << error;
}

// Ensure the COO matrix carries an explicit data array. When the COO has no
// explicit data, Ascend COOToCSR (coo2csr.cc) renumbers the CSR data array in
// CSR-sorted order instead of carrying the original COO edge positions. This
// breaks eid semantics for downstream ops (COOGetRowDataAndIndices /
// COOSliceRows / COOSliceMatrix) whose results depend on eid == original COO
// position.
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
  CHECK(row >= 0 && row < coo.num_rows) << "Invalid row index: " << row;
  CHECK(coo.row->ctx.device_type == kDGLAscend)
      << "COOGetRowDataAndIndices expects Ascend input";
  CHECK_EQ(coo.row->ndim, 1) << "COO row must be a 1D array";
  CHECK_EQ(coo.col->ndim, 1) << "COO col must be a 1D array";
  CHECK(coo.row.IsContiguous()) << "COO row must be contiguous";
  CHECK(coo.col.IsContiguous()) << "COO col must be contiguous";
  CHECK_EQ(coo.row->shape[0], coo.col->shape[0])
      << "COO row and col must have the same length";
  CHECK_EQ(coo.row->dtype.code, kDGLInt) << "COO IDs must be integers";
  CHECK_EQ(coo.row->dtype.bits, sizeof(IdType) * 8)
      << "COO ID dtype does not match dispatch type";
  CHECK_EQ(coo.row->dtype.lanes, 1) << "COO IDs must be scalar integers";
  CHECK_EQ(coo.col->dtype.code, coo.row->dtype.code);
  CHECK_EQ(coo.col->dtype.bits, coo.row->dtype.bits);
  CHECK_EQ(coo.col->dtype.lanes, coo.row->dtype.lanes);
  CHECK_EQ(coo.col->ctx.device_type, coo.row->ctx.device_type);
  CHECK_EQ(coo.col->ctx.device_id, coo.row->ctx.device_id);

  const bool has_data = COOHasData(coo);
  if (has_data) {
    CHECK_EQ(coo.data->ndim, 1) << "COO data must be a 1D array";
    CHECK(coo.data.IsContiguous()) << "COO data must be contiguous";
    CHECK_EQ(coo.data->shape[0], coo.row->shape[0])
        << "COO data must have the same length as row and col";
    CHECK_EQ(coo.data->dtype.code, coo.row->dtype.code);
    CHECK_EQ(coo.data->dtype.bits, coo.row->dtype.bits);
    CHECK_EQ(coo.data->dtype.lanes, coo.row->dtype.lanes);
    CHECK_EQ(coo.data->ctx.device_type, coo.row->ctx.device_type);
    CHECK_EQ(coo.data->ctx.device_id, coo.row->ctx.device_id);
  }

  CHECK_LE(row, static_cast<int64_t>(std::numeric_limits<IdType>::max()))
      << "Target row cannot be represented by the COO ID dtype";
  const int64_t nnz = coo.row->shape[0];
  CHECK_GE(nnz, 0) << "COO nnz must be non-negative";
  CHECK_LE(
      static_cast<uint64_t>(nnz),
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
      << "COOGetRowDataAndIndices supports at most UINT32_MAX edges";
  if (!has_data && nnz != 0) {
    CHECK_LE(
        static_cast<uint64_t>(nnz - 1),
        static_cast<uint64_t>(std::numeric_limits<IdType>::max()))
        << "Implicit COO edge IDs cannot be represented by the ID dtype";
  }

  const DGLContext ctx = coo.row->ctx;
  const uint8_t id_bits = coo.row->dtype.bits;
  if (nnz == 0) {
    return std::make_pair(
        NewIdArray(0, ctx, id_bits), NewIdArray(0, ctx, id_bits));
  }

  const uint32_t nnz_u32 = static_cast<uint32_t>(nnz);
  const uint32_t block_dim =
      std::min<uint32_t>(kCooGetRowMaxBlocks, nnz_u32);
  CHECK_GE(block_dim, 1U);
  CHECK_LE(block_dim, kCooGetRowMaxBlocks);

  CheckCooGetRowAcl(
      aclrtSetDevice(ctx.device_id), "COOGetRowDataAndIndices set device");
  aclrtStream stream = dgl::runtime::getCurrentAscendStream();

  CooGetRowWorkspaces workspaces;
  workspaces.Allocate(block_dim, sizeof(CooGetRowTiling<IdType>));

  CooGetRowTiling<IdType> tiling_host{};
  tiling_host.nnz = nnz_u32;
  tiling_host.target_row = static_cast<IdType>(row);
  tiling_host.has_data = has_data ? 1U : 0U;
  CheckCooGetRowAcl(
      aclrtMemcpy(
          workspaces.tiling, sizeof(CooGetRowTiling<IdType>), &tiling_host,
          sizeof(CooGetRowTiling<IdType>), ACL_MEMCPY_HOST_TO_DEVICE),
      "COOGetRowDataAndIndices tiling copy");

  LaunchCooGetRowCount<IdType>(
      block_dim, stream, coo.row->data, workspaces.count_slabs,
      workspaces.tiling);
  CheckCooGetRowAcl(
      aclrtSynchronizeStream(stream),
      "COOGetRowDataAndIndices count synchronization");

  const size_t count_words =
      static_cast<size_t>(block_dim) * kCooGetRowCountStrideWords;
  const size_t count_bytes = count_words * sizeof(uint32_t);
  std::vector<uint32_t> count_slabs_host(count_words);
  CheckCooGetRowAcl(
      aclrtMemcpy(
          count_slabs_host.data(), count_bytes, workspaces.count_slabs,
          count_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
      "COOGetRowDataAndIndices count copy");

  std::vector<uint32_t> offsets_host(
      static_cast<size_t>(block_dim) + 1, 0);
  uint64_t total_matches = 0;
  const uint32_t per_block = nnz_u32 / block_dim;
  const uint32_t remainder = nnz_u32 % block_dim;
  for (uint32_t block = 0; block < block_dim; ++block) {
    const uint32_t count =
        count_slabs_host[static_cast<size_t>(block) *
                         kCooGetRowCountStrideWords];
    const uint32_t block_length = per_block + (block < remainder ? 1U : 0U);
    CHECK_LE(count, block_length)
        << "COOGetRowDataAndIndices count exceeds its block range";
    CHECK_LE(
        total_matches + count,
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
        << "COOGetRowDataAndIndices prefix sum exceeds uint32 capacity";
    total_matches += count;
    offsets_host[static_cast<size_t>(block) + 1] =
        static_cast<uint32_t>(total_matches);
  }
  CHECK_LE(total_matches, static_cast<uint64_t>(nnz_u32))
      << "COOGetRowDataAndIndices matched more edges than the COO contains";

  if (total_matches == 0) {
    workspaces.Free();
    return std::make_pair(
        NewIdArray(0, ctx, id_bits), NewIdArray(0, ctx, id_bits));
  }

  CHECK_LE(
      total_matches,
      static_cast<uint64_t>(
          std::numeric_limits<size_t>::max() / sizeof(IdType)))
      << "COOGetRowDataAndIndices output byte size overflows size_t";
  NDArray data =
      NewIdArray(static_cast<int64_t>(total_matches), ctx, id_bits);
  NDArray col_indices =
      NewIdArray(static_cast<int64_t>(total_matches), ctx, id_bits);

  const size_t offset_bytes =
      offsets_host.size() * sizeof(offsets_host[0]);
  CheckCooGetRowAcl(
      aclrtMemcpy(
          workspaces.offsets, offset_bytes, offsets_host.data(), offset_bytes,
          ACL_MEMCPY_HOST_TO_DEVICE),
      "COOGetRowDataAndIndices offsets copy");

  LaunchCooGetRowScatter<IdType>(
      block_dim, stream, coo.row->data, coo.col->data,
      has_data ? coo.data->data : nullptr, workspaces.offsets, data->data,
      col_indices->data, workspaces.tiling);
  CheckCooGetRowAcl(
      aclrtSynchronizeStream(stream),
      "COOGetRowDataAndIndices scatter synchronization");
  workspaces.Free();
  return std::make_pair(data, col_indices);
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
