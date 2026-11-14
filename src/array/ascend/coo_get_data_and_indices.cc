#ifdef DGL_USE_ASCEND

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <dgl/array.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "../array_op.h"
#include "../../runtime/workspace.h"

extern "C" uint32_t aclrtlaunch_coo_getData_and_indices(
    uint32_t blockDim, aclrtStream stream, uint32_t operation,
    void* cooRows, void* cooCols,
    void* queryRows, void* queryCols,
    void* cooData, void* mask, void* offsets,
    void* outRows, void* outCols, void* outData, void* tiling);

namespace dgl {
namespace runtime {

aclrtStream getCurrentAscendStream();

}  // namespace runtime

namespace aten {
namespace impl {
namespace {

constexpr uint32_t TileLength = 256;
constexpr uint32_t DefaultMatchesPerQuery = 8;
constexpr uint32_t MatchCountSlotElements = 32 / sizeof(uint64_t);

// Keep the field order identical to the kernel-side structure.
struct COOGetDataAndIndicesTiling {
  uint32_t nnz;
  uint32_t tileCount;
  uint32_t queryCount;
  uint32_t hasData;
};

inline void CheckAcl(aclError status, const char* operation) {
  CHECK_EQ(status, ACL_SUCCESS)
      << operation << " failed, aclError=" << status;
}

inline size_t QueryBytes(uint64_t count, size_t element_bytes) {
  CHECK_LE(count, static_cast<uint64_t>(std::numeric_limits<size_t>::max() / element_bytes)) << "COOGetDataAndIndices byte size overflow";
  return static_cast<size_t>(count) * element_bytes;
}

template <typename IdType>
std::vector<NDArray> COOGetDataAndIndicesAscend(
    COOMatrix coo, NDArray rows, NDArray cols) {
  const DGLContext ctx = coo.row->ctx;

  const int64_t nnz = coo.row->shape[0];
  const bool hasData = COOHasData(coo);
  const int64_t rowlen = rows->shape[0];
  const int64_t collen = cols->shape[0];
  const int64_t len = rowlen;

  CHECK_EQ(rowlen, collen) << "Query rows and cols must have the same length.";

  auto empty_result = [&]() -> std::vector<NDArray> {
    return {
        NDArray::Empty({0}, coo.row->dtype, ctx),
        NDArray::Empty({0}, coo.row->dtype, ctx),
        NDArray::Empty({0}, coo.row->dtype, ctx)};
  };

  if (rowlen == 0 || nnz == 0) {
    return empty_result();
  }

  const uint64_t queryCount = static_cast<uint64_t>(len);
  // 处理 nnz 个元素需要多少个 tile
  const uint64_t tileCount = (static_cast<uint64_t>(nnz) + TileLength - 1) / TileLength;

  CheckAcl(aclrtSetDevice(ctx.device_id), "aclrtSetDevice");
  const aclrtStream stream = dgl::runtime::getCurrentAscendStream();

  uint32_t blockDim = 1;
  int64_t coreCount = 0;
  const aclError core_status = aclrtGetDeviceInfo(
      ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreCount);
  if (core_status == ACL_SUCCESS &&
      coreCount > 0 && coreCount <= 4096) {
    blockDim = static_cast<uint32_t>(
        std::min<uint64_t>(
            queryCount, static_cast<uint64_t>(coreCount)));
  }

  // 分配 match 空间
  const uint64_t resultElements = queryCount * DefaultMatchesPerQuery;
  const size_t resultBytes = QueryBytes(resultElements, sizeof(IdType));
  const size_t offsetBytes = QueryBytes(static_cast<uint64_t>(blockDim) + 1, sizeof(uint64_t));
  // 记录每个 block match 的数量
  const uint64_t matchCountElements = static_cast<uint64_t>(blockDim) * MatchCountSlotElements;
  const size_t matchCountBytes = QueryBytes(matchCountElements, sizeof(uint64_t));

  std::vector<uint64_t> offsets(static_cast<size_t>(blockDim) + 1, 0);
  std::vector<uint64_t> matchCounts(static_cast<size_t>(matchCountElements), 0);

  COOGetDataAndIndicesTiling tiling{
      static_cast<uint32_t>(nnz),
      static_cast<uint32_t>(tileCount),
      static_cast<uint32_t>(queryCount),
      hasData ? 1U : 0U};

  auto* device = dgl::runtime::DeviceAPI::Get(ctx);
  dgl::runtime::Workspace<void> matchRowsWorkspace(device, ctx, resultBytes);
  dgl::runtime::Workspace<void> matchColsWorkspace(device, ctx, resultBytes);
  dgl::runtime::Workspace<void> matchDataWorkspace(device, ctx, resultBytes);
  dgl::runtime::Workspace<void> offsetsWorkspace(device, ctx, offsetBytes);
  dgl::runtime::Workspace<void> matchCountsWorkspace(device, ctx, matchCountBytes);
  dgl::runtime::Workspace<void> tilingWorkspace(device, ctx, sizeof(tiling));

  void* matchRowsDev = matchRowsWorkspace.get();
  void* matchColsDeV = matchColsWorkspace.get();
  void* matchDataDeV = matchDataWorkspace.get();
  void* offsetsDeV = offsetsWorkspace.get();
  void* matchCountsDeV = matchCountsWorkspace.get();
  void* tilingDeV = tilingWorkspace.get();
  void* cooDataDeV = hasData ? coo.data->data : nullptr;

  // Operations 0/1 match int32/int64; operations 2/3 gather int32/int64.
  const uint32_t type_operation = std::is_same<IdType, int32_t>::value ? 0U : 1U;

  CheckAcl(aclrtMemcpy(tilingDeV, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy tiling H2D");
  const aclError match_status = aclrtlaunch_coo_getData_and_indices(
      blockDim, stream, type_operation,
      coo.row->data, coo.col->data,
      rows->data, cols->data,
      cooDataDeV, matchRowsDev, matchCountsDeV,
      matchColsDeV, matchDataDeV, nullptr, tilingDeV);
  CheckAcl(match_status, "COOGetDataAndIndices match launch");
  CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream after match");
  CheckAcl(aclrtMemcpy(matchCounts.data(), matchCountBytes, matchCountsDeV, matchCountBytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy match counts D2H");

  for (uint32_t block = 0; block < blockDim; ++block) {
    const uint64_t query_begin = queryCount * block / blockDim;
    const uint64_t query_end = queryCount * (block + 1) / blockDim;
    const uint64_t default_capacity = (query_end - query_begin) * DefaultMatchesPerQuery;
    const uint64_t count = matchCounts[static_cast<uint64_t>(block) * MatchCountSlotElements];
    offsets[block + 1] = offsets[block] + count;
  }

  const uint64_t total = offsets.back();
  QueryBytes(total, sizeof(IdType));

  if (total == 0) {
    return empty_result();
  }

  CheckAcl(aclrtMemcpy(offsetsDeV, offsetBytes, offsets.data(), offsetBytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy block offsets H2D");

  const int64_t output_length = static_cast<int64_t>(total);
  NDArray outRows = NDArray::Empty({output_length}, coo.row->dtype, ctx);
  NDArray outCols = NDArray::Empty({output_length}, coo.row->dtype, ctx);
  NDArray outData = NDArray::Empty({output_length}, coo.row->dtype, ctx);

  const aclError gather_status = aclrtlaunch_coo_getData_and_indices(
      blockDim, stream, type_operation + 2U,
      matchRowsDev, matchColsDeV, nullptr, nullptr,
      matchDataDeV, nullptr, offsetsDeV,
      outRows->data, outCols->data, outData->data, tilingDeV);

  CheckAcl(gather_status, "COOGetDataAndIndices gather launch");
  CheckAcl(aclrtSynchronizeStream(stream),"aclrtSynchronizeStream after gather");
  return {outRows, outCols, outData};
}

}  // namespace

template <>
std::vector<NDArray>
COOGetDataAndIndices<kDGLAscend, int32_t>(
    COOMatrix coo, NDArray rows, NDArray cols) {
  return COOGetDataAndIndicesAscend<int32_t>(
      coo, rows, cols);
}

template <>
std::vector<NDArray>
COOGetDataAndIndices<kDGLAscend, int64_t>(
    COOMatrix coo, NDArray rows, NDArray cols) {
  return COOGetDataAndIndicesAscend<int64_t>(
      coo, rows, cols);
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_USE_ASCEND
