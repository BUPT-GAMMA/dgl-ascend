/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_sampling_uniform.cc
 * @brief Ascend host launcher for uniform CSR row-wise sampling.
 *
 * Single-core (blockDim=1) native AscendC kernel path — verified correct.
 */

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e;              \
  }

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_uniform_int32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data, void* rows,
    void* out_ptr, void* out_rows, void* out_cols, void* out_idxs,
    void* tiling);

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_uniform_int64(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data, void* rows,
    void* out_ptr, void* out_rows, void* out_cols, void* out_idxs,
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

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(
    CSRMatrix mat, IdArray rows, int64_t num_samples, bool replace) {
#ifdef DGL_USE_ASCEND
  auto ctx = mat.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend)
      << "Expected Ascend device context for CSRRowWiseSamplingUniform";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  const bool select_all = (num_samples == -1);
  replace = (replace && !select_all);

  const int64_t num_rows = rows->shape[0];
  const uint8_t nbits = mat.indptr->dtype.bits;

  if (num_rows == 0 || (num_samples == 0 && !select_all)) {
    IdArray empty_row = aten::NewIdArray(0, ctx, nbits);
    return COOMatrix(mat.num_rows, mat.num_cols, empty_row, empty_row,
                     empty_row);
  }

  int64_t max_output;
  if (select_all) {
    NDArray deg = CSRGetRowNNZ<kDGLAscend, IdType>(mat, rows);
    std::vector<IdType> deg_host(num_rows);
    ASCEND_CALL(aclrtMemcpy(deg_host.data(), num_rows * sizeof(IdType),
        deg->data, num_rows * sizeof(IdType),
        ACL_MEMCPY_DEVICE_TO_HOST));
    IdType sum_deg = 0;
    for (int64_t i = 0; i < num_rows; ++i) sum_deg += deg_host[i];
    max_output = static_cast<int64_t>(sum_deg);
  } else {
    // Saturating product: overflow of int64 (or of the IdType index space)
    // must fail loudly, not silently wrap and under-allocate the output.
    constexpr int64_t kMaxInt64 = std::numeric_limits<int64_t>::max();
    CHECK(num_rows <= kMaxInt64 / num_samples)
        << "num_rows * num_samples overflows int64: " << num_rows << " * "
        << num_samples;
    max_output = num_rows * num_samples;
    CHECK(max_output <= static_cast<int64_t>(std::numeric_limits<IdType>::max()))
        << "Output size " << max_output << " exceeds IdType range";
  }

  IdArray picked_row = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_col = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_idx = aten::NewIdArray(max_output, ctx, nbits);
  IdArray out_ptr = aten::NewIdArray(num_rows + 1, ctx, nbits);

  if (max_output == 0) {
    return COOMatrix(mat.num_rows, mat.num_cols,
                     picked_row.CreateView({0}, picked_row->dtype),
                     picked_col.CreateView({0}, picked_col->dtype),
                     picked_idx.CreateView({0}, picked_idx->dtype));
  }

  auto stream = dgl::runtime::getCurrentAscendStream();
  const bool has_data = aten::CSRHasData(mat);
  void* data_ptr = has_data ? mat.data->data : nullptr;

  uint32_t tiling_data[7] = {
      static_cast<uint32_t>(num_rows),
      select_all ? 0u : static_cast<uint32_t>(num_samples),
      static_cast<uint32_t>(replace ? 1 : 0),
      static_cast<uint32_t>(has_data ? 1 : 0),
      static_cast<uint32_t>(
          RandomEngine::ThreadLocal()->RandInt(1000000000)),
      static_cast<uint32_t>(select_all ? 1 : 0),
      static_cast<uint32_t>(mat.num_rows),
  };
  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev, sizeof(tiling_data),
                          ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tiling_dev, sizeof(tiling_data), tiling_data,
                          sizeof(tiling_data), ACL_MEMCPY_HOST_TO_DEVICE));

  const uint32_t block_dim = 1;
  if (std::is_same<IdType, int32_t>::value) {
    aclError err = aclrtlaunch_csr_row_wise_sampling_uniform_int32(
        block_dim, stream, mat.indptr->data, mat.indices->data, data_ptr,
        rows->data, out_ptr->data, picked_row->data, picked_col->data,
        picked_idx->data, tiling_dev);
    CHECK(err == ACL_SUCCESS)
        << "csr_row_wise_sampling_uniform_int32 launch failed: " << err;
  } else {
    aclError err = aclrtlaunch_csr_row_wise_sampling_uniform_int64(
        block_dim, stream, mat.indptr->data, mat.indices->data, data_ptr,
        rows->data, out_ptr->data, picked_row->data, picked_col->data,
        picked_idx->data, tiling_dev);
    CHECK(err == ACL_SUCCESS)
        << "csr_row_wise_sampling_uniform_int64 launch failed: " << err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tiling_dev));

  IdType total = 0;
  ASCEND_CALL(aclrtMemcpy(&total, sizeof(IdType),
      static_cast<char*>(out_ptr->data) + num_rows * sizeof(IdType),
      sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
  const int64_t new_len = static_cast<int64_t>(total);

  return COOMatrix(
      mat.num_rows, mat.num_cols,
      picked_row.CreateView({new_len}, picked_row->dtype),
      picked_col.CreateView({new_len}, picked_col->dtype),
      picked_idx.CreateView({new_len}, picked_idx->dtype));
#else
  LOG(FATAL) << "Ascend support is not compiled. "
                "Please compile with -DUSE_ASCEND=ON";
  return {};
#endif  // DGL_USE_ASCEND
}

template COOMatrix CSRRowWiseSamplingUniform<kDGLAscend, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDGLAscend, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
