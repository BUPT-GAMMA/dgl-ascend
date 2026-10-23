
#ifdef DGL_USE_ASCEND

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <dgl/array.h>
#include <dgl/aten/array_ops.h>

#include "../array_op.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" uint32_t aclrtlaunch_coo_get_row_nnz_int32(
    uint32_t blockDim,
    aclrtStream stream,
    void* cooRows,
    void* queries,
    void* output,
    void* tiling);

extern "C" uint32_t aclrtlaunch_coo_get_row_nnz_int64(
    uint32_t blockDim,
    aclrtStream stream,
    void* cooRows,
    void* queries,
    void* output,
    void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}
}

namespace dgl {
namespace aten {
namespace impl {

namespace {

#define COO_ASCEND_CHECK(expr)                    \
  do {                                            \
    aclError err = (expr);                        \
    CHECK(err == ACL_SUCCESS)                     \
        << "COOGetRowNNZ ACL error: " << err;      \
  } while (0)

struct TilingBuffer {
  void* ptr = nullptr;

  ~TilingBuffer() {
    if (ptr != nullptr) {
      aclrtFree(ptr);
    }
  }
};

template <typename T>
void ValidateQueries(
    NDArray queries, int64_t numRows) {

  DGLContext cpuCtx{kDGLCPU, 0};

  // Validate on CPU to preserve the original
  // COOGetRowNNZ invalid-row behavior.
  auto cpuQueries = queries.CopyTo(cpuCtx);

  auto values =
      cpuQueries.template ToVector<T>();

  for (T row : values) {
    CHECK(row >= 0 &&
          static_cast<int64_t>(row) < numRows)
        << "Invalid row index: " << row;
  }
}

template <typename T>
NDArray LaunchBatch(
    COOMatrix coo, NDArray queries) {

  auto ctx = coo.row->ctx;

  CHECK(ctx.device_type == kDGLAscend);
  CHECK(queries->ctx.device_type == kDGLAscend);
  CHECK_EQ(queries->ctx.device_id, ctx.device_id);

  CHECK_SAME_DTYPE(coo.row, queries);
  CHECK_SAME_DTYPE(coo.col, queries);

  const int64_t nnz = coo.row->shape[0];
  const int64_t count = queries->shape[0];

  NDArray output =
      NDArray::Empty({count}, queries->dtype, ctx);

  if (count == 0) {
    return output;
  }

  COO_ASCEND_CHECK(
      aclrtSetDevice(ctx.device_id));

  auto stream =
      dgl::runtime::getCurrentAscendStream();

  // Ensure any earlier NPU writes are visible
  // before copying query indices to the CPU.
  COO_ASCEND_CHECK(
      aclrtSynchronizeStream(stream));

  ValidateQueries<T>(queries, coo.num_rows);

  uint64_t tilingData[2] = {
      static_cast<uint64_t>(nnz),
      static_cast<uint64_t>(count)
  };

  TilingBuffer tiling;

  COO_ASCEND_CHECK(
      aclrtMalloc(
          &tiling.ptr,
          sizeof(tilingData),
          ACL_MEM_MALLOC_HUGE_FIRST));

  COO_ASCEND_CHECK(
      aclrtMemcpy(
          tiling.ptr,
          sizeof(tilingData),
          tilingData,
          sizeof(tilingData),
          ACL_MEMCPY_HOST_TO_DEVICE));

  // Each block owns a 64-byte-aligned
  // range of query results.
  constexpr uint64_t cacheLine = 64;

  uint64_t alignElems =
      cacheLine / sizeof(T);

  uint64_t groups =
      1 + (static_cast<uint64_t>(count) - 1)
              / alignElems;

  uint32_t blockDim =
      static_cast<uint32_t>(
          std::min<uint64_t>(groups, 8));

  uint32_t err = 0;

  if constexpr (sizeof(T) == 4) {
    err = aclrtlaunch_coo_get_row_nnz_int32(
        blockDim,
        stream,
        coo.row->data,
        queries->data,
        output->data,
        tiling.ptr);
  } else {
    err = aclrtlaunch_coo_get_row_nnz_int64(
        blockDim,
        stream,
        coo.row->data,
        queries->data,
        output->data,
        tiling.ptr);
  }

  CHECK(err == ACL_SUCCESS)
      << "COOGetRowNNZ launch failed: " << err;

  COO_ASCEND_CHECK(
      aclrtSynchronizeStream(stream));

  return output;
}

}  // namespace

// Batch query: returns Ascend NDArray.
NDArray COOGetRowNNZDirect(
    COOMatrix coo, NDArray queries) {

  if (coo.row->dtype.bits == 32) {
    return LaunchBatch<int32_t>(coo, queries);
  }

  CHECK_EQ(coo.row->dtype.bits, 64);
  return LaunchBatch<int64_t>(coo, queries);
}

// Scalar query: returns int64_t on Host.
int64_t COOGetRowNNZDirect(
    COOMatrix coo, int64_t row) {

  CHECK(row >= 0 && row < coo.num_rows)
      << "Invalid row index: " << row;

  DGLContext cpuCtx{kDGLCPU, 0};
  auto ascendCtx = coo.row->ctx;

  if (coo.row->dtype.bits == 32) {

    // An int32 COO cannot store row indices
    // exceeding int32_t's maximum.
    if (row > std::numeric_limits<int32_t>::max()) {
      return 0;
    }

    std::vector<int32_t> values = {
        static_cast<int32_t>(row)
    };

    auto cpuQuery = VecToIdArray<int32_t>(
        values, 32, cpuCtx);

    auto npuQuery = cpuQuery.CopyTo(ascendCtx);

    auto npuResult =
        LaunchBatch<int32_t>(coo, npuQuery);

    auto result =
        npuResult.CopyTo(cpuCtx)
            .ToVector<int32_t>();

    return static_cast<int64_t>(result[0]);
  }

  CHECK_EQ(coo.row->dtype.bits, 64);

  std::vector<int64_t> values = {row};

  auto cpuQuery = VecToIdArray<int64_t>(
      values, 64, cpuCtx);

  auto npuQuery = cpuQuery.CopyTo(ascendCtx);

  auto npuResult =
      LaunchBatch<int64_t>(coo, npuQuery);

  auto result =
      npuResult.CopyTo(cpuCtx)
          .ToVector<int64_t>();

  return result[0];
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_USE_ASCEND
