
#ifdef DGL_USE_ASCEND

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <dgl/array.h>
#include <dgl/aten/array_ops.h>

#include "../array_op.h"
#include "../../runtime/workspace.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" uint32_t aclrtlaunch_coo_get_row_nnz_int32(uint32_t blockDim, aclrtStream stream, void* cooRows, void* queries, void* output, void* tiling);

extern "C" uint32_t aclrtlaunch_coo_get_row_nnz_int64(uint32_t blockDim, aclrtStream stream, void* cooRows, void* queries, void* output, void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}
}

namespace dgl {
namespace aten {
namespace impl {

namespace {

// Keep the field order identical to the kernel-side structure.
struct COOGetRowNNZTiling {
  uint64_t nnz;
  uint64_t queryCount;
};

inline void CheckAcl(aclError status, const char* operation) {
  CHECK_EQ(status, ACL_SUCCESS)
      << operation << " failed, aclError=" << status;
}

template <typename T>
NDArray COOGetRowNNZBatch(COOMatrix coo, NDArray queries) {

  auto ctx = coo.row->ctx;
  const int64_t nnz = coo.row->shape[0];
  const int64_t count = queries->shape[0];
  NDArray output = NDArray::Empty({count}, queries->dtype, ctx);

  if (count == 0) {
    return output;
  }

  CheckAcl(aclrtSetDevice(ctx.device_id), "aclrtSetDevice");
  auto stream = dgl::runtime::getCurrentAscendStream();
  COOGetRowNNZTiling tiling{
      static_cast<uint64_t>(nnz),
      static_cast<uint64_t>(count)};

  auto* device = dgl::runtime::DeviceAPI::Get(ctx);
  dgl::runtime::Workspace<void> tilingWorkspace(device, ctx, sizeof(tiling));
  void* tilingDev = tilingWorkspace.get();

  CheckAcl(aclrtMemcpy(tilingDev, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy tiling H2D");

  uint32_t blockDim = 1;
  int64_t coreCount = 0;
  const aclError core_status = aclrtGetDeviceInfo(ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreCount);
  if (core_status == ACL_SUCCESS &&
      coreCount > 0 && coreCount <= 4096) {
    blockDim = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(count), static_cast<uint64_t>(coreCount)));
  }

  uint32_t err = 0;
  if constexpr (sizeof(T) == 4) {
    err = aclrtlaunch_coo_get_row_nnz_int32(blockDim, stream, coo.row->data, queries->data, output->data, tilingDev);
  } else {
    err = aclrtlaunch_coo_get_row_nnz_int64(blockDim, stream, coo.row->data, queries->data, output->data, tilingDev);
  }
  CHECK(err == ACL_SUCCESS) << "COOGetRowNNZ launch failed: " << err;
  CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream after kernel");
  return output;
}

}  // namespace

NDArray COOGetRowNNZAscend(COOMatrix coo, NDArray queries) {
  if (coo.row->dtype.bits == 32) {
    return COOGetRowNNZBatch<int32_t>(coo, queries);
  }else{
    return COOGetRowNNZBatch<int64_t>(coo, queries);
  }

}
int64_t COOGetRowNNZAscend(COOMatrix coo, int64_t row) {

  DGLContext cpuCtx{kDGLCPU, 0};
  auto ascendCtx = coo.row->ctx;

  if (coo.row->dtype.bits == 32) {
    std::vector<int32_t> values = {static_cast<int32_t>(row)};
    auto cpuQuery = VecToIdArray<int32_t>(values, 32, cpuCtx);
    auto npuQuery = cpuQuery.CopyTo(ascendCtx);
    auto npuResult =COOGetRowNNZBatch<int32_t>(coo, npuQuery);
    auto result = npuResult.CopyTo(cpuCtx).ToVector<int32_t>();
    return static_cast<int64_t>(result[0]);
  }else{
    std::vector<int64_t> values = {row};
    auto cpuQuery = VecToIdArray<int64_t>(values, 64, cpuCtx);
    auto npuQuery = cpuQuery.CopyTo(ascendCtx);
    auto npuResult = COOGetRowNNZBatch<int64_t>(coo, npuQuery);
    auto result =npuResult.CopyTo(cpuCtx).ToVector<int64_t>();
    return result[0];
  }
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_USE_ASCEND
