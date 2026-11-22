/**
 * @file src/array/ascend/coo_ops.cc
 * @brief Ascend NPU implementations of COO operators.
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
#include "../../runtime/workspace.h"

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

void CheckAcl(aclError error, const char* operation) {
  CHECK(error == ACL_SUCCESS) << operation << " failed with Ascend error code " << error;
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
std::pair<NDArray, NDArray> COOGetRowDataAndIndicesAscend(COOMatrix coo, int64_t row) {
  const bool hasData = COOHasData(coo);
  const int64_t nnz64 = coo.row->shape[0];
  const DGLContext ctx = coo.row->ctx;
  const uint8_t id_bits = coo.row->dtype.bits;
  if (nnz64 == 0) {
    return std::make_pair(NewIdArray(0, ctx, id_bits), NewIdArray(0, ctx, id_bits));
  }

  const uint32_t nnz = static_cast<uint32_t>(nnz64);
  uint32_t blockDim = 1;
  int64_t coreCount = 0;
  const aclError core_status = aclrtGetDeviceInfo(
      ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreCount);
  if (core_status == ACL_SUCCESS && coreCount > 0 && coreCount <= 4096) {
    blockDim = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(nnz), static_cast<uint64_t>(coreCount)));
  }

  CheckAcl(aclrtSetDevice(ctx.device_id), "COOGetRowDataAndIndices set device");
  aclrtStream stream = dgl::runtime::getCurrentAscendStream();
  const size_t countBytes = static_cast<size_t>(blockDim) * kCooGetRowCountWords * sizeof(uint32_t);
  const size_t offsetBytes = (static_cast<size_t>(blockDim) + 1) * sizeof(uint32_t);
  auto* device = dgl::runtime::DeviceAPI::Get(ctx);
  dgl::runtime::Workspace<void> countsWorkspace(device, ctx, countBytes);
  dgl::runtime::Workspace<void> offsetsWorkspace(device, ctx, offsetBytes);
  dgl::runtime::Workspace<void> tilingWorkspace(device, ctx, sizeof(CooGetRowTiling<IdType>));

  void* countsDev = countsWorkspace.get();
  void* offsetsDev = offsetsWorkspace.get();
  void* tilingDev = tilingWorkspace.get();

  CooGetRowTiling<IdType> tilingHost{};
  tilingHost.nnz = nnz;
  tilingHost.targetRow = static_cast<IdType>(row);
  tilingHost.hasData = hasData ? 1U : 0U;
  CheckAcl(aclrtMemcpy(tilingDev, sizeof(CooGetRowTiling<IdType>), &tilingHost, sizeof(CooGetRowTiling<IdType>), ACL_MEMCPY_HOST_TO_DEVICE), "COOGetRowDataAndIndices tiling copy");
  
  uint32_t countLaunchError = ACL_SUCCESS;
  if (std::is_same<IdType, int32_t>::value) {
    countLaunchError = aclrtlaunch_coo_get_row_count_i32(blockDim, stream, coo.row->data, countsDev, tilingDev);
  } else {
    countLaunchError = aclrtlaunch_coo_get_row_count_i64(blockDim, stream, coo.row->data, countsDev, tilingDev);
  }
  CHECK(countLaunchError == ACL_SUCCESS) << "COOGetRowDataAndIndices count launch failed: " << countLaunchError;
  CheckAcl(aclrtSynchronizeStream(stream), "COOGetRowDataAndIndices count synchronization");

  const size_t countWords = static_cast<size_t>(blockDim) * kCooGetRowCountWords;
  std::vector<uint32_t> countsHost(countWords);
  CheckAcl(aclrtMemcpy(countsHost.data(), countBytes, countsDev, countBytes, ACL_MEMCPY_DEVICE_TO_HOST), "COOGetRowDataAndIndices count copy");

  std::vector<uint32_t> offsetsHost(static_cast<size_t>(blockDim) + 1, 0);
  uint64_t totalMatches = 0;
  const uint32_t perBlock = nnz / blockDim;
  const uint32_t remainder = nnz % blockDim;
  for (uint32_t block = 0; block < blockDim; ++block) {
    const uint32_t count = countsHost[static_cast<size_t>(block) * kCooGetRowCountWords];
    const uint32_t blockLength = perBlock + (block < remainder ? 1U : 0U);
    totalMatches += count;
    offsetsHost[static_cast<size_t>(block) + 1] = static_cast<uint32_t>(totalMatches);
  }

  if (totalMatches == 0) {
    return std::make_pair(NewIdArray(0, ctx, id_bits), NewIdArray(0, ctx, id_bits));
  }

  NDArray data = NewIdArray(static_cast<int64_t>(totalMatches), ctx, id_bits);
  NDArray colIndices = NewIdArray(static_cast<int64_t>(totalMatches), ctx, id_bits);
  CheckAcl(aclrtMemcpy(offsetsDev, offsetBytes, offsetsHost.data(), offsetBytes, ACL_MEMCPY_HOST_TO_DEVICE), "COOGetRowDataAndIndices offsets copy");

  uint32_t scatterLaunchError = ACL_SUCCESS;
  if (std::is_same<IdType, int32_t>::value) {
    scatterLaunchError = aclrtlaunch_coo_get_row_scatter_i32(blockDim, stream, coo.row->data, coo.col->data, hasData ? coo.data->data : nullptr, offsetsDev, data->data, colIndices->data, tilingDev);
  } else {
    scatterLaunchError = aclrtlaunch_coo_get_row_scatter_i64(blockDim, stream, coo.row->data, coo.col->data, hasData ? coo.data->data : nullptr, offsetsDev, data->data, colIndices->data, tilingDev);
  }
  CHECK(scatterLaunchError == ACL_SUCCESS) << "COOGetRowDataAndIndices scatter launch failed: " << scatterLaunchError;
  CheckAcl(aclrtSynchronizeStream(stream), "COOGetRowDataAndIndices scatter synchronization");
  return std::make_pair(data, colIndices);
}

template <>
std::pair<NDArray, NDArray> COOGetRowDataAndIndices<kDGLAscend, int32_t>( COOMatrix coo, int64_t row) {
  return COOGetRowDataAndIndicesAscend<int32_t>(coo, row);
}

template <>
std::pair<NDArray, NDArray> COOGetRowDataAndIndices<kDGLAscend, int64_t>(COOMatrix coo, int64_t row) {
  return COOGetRowDataAndIndicesAscend<int64_t>(coo, row);
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
