#ifdef DGL_USE_ASCEND

#ifdef DGL_USE_PYTORCH_NPU_STREAM
// Include before DGL headers so dmlc retains ownership of LOG/CHECK macros.
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "../../runtime/workspace.h"
#include "sddmm_csr.h"

// Both launcher families share one ABI. Add keeps its compile-time fast path;
// the generic family selects sub/mul/div/copy/dot from the shared tiling data.
#define SDDMMCSR_DECLARE_ENTRY(Name) \
  extern "C" uint32_t aclrtlaunch_##Name( \
      uint32_t, aclrtStream, void*, void*, void*, void*, void*, void*, \
      void*, void*, void*);
#define SDDMMCSR_DECLARE(Key, Lhs, Rhs, Bcast, Map) \
  SDDMMCSR_DECLARE_ENTRY(sddmm_csr_add_f32_i32_##Key) \
  SDDMMCSR_DECLARE_ENTRY(sddmm_csr_add_f32_i64_##Key) \
  SDDMMCSR_DECLARE_ENTRY(sddmm_csr_add_f64_i32_##Key) \
  SDDMMCSR_DECLARE_ENTRY(sddmm_csr_add_f64_i64_##Key) \
  SDDMMCSR_DECLARE_ENTRY(sddmm_csr_f32_i32_##Key) \
  SDDMMCSR_DECLARE_ENTRY(sddmm_csr_f32_i64_##Key) \
  SDDMMCSR_DECLARE_ENTRY(sddmm_csr_f64_i32_##Key) \
  SDDMMCSR_DECLARE_ENTRY(sddmm_csr_f64_i64_##Key)
SDDMMCSR_FOR_EACH_KEY(SDDMMCSR_DECLARE)
#undef SDDMMCSR_DECLARE
#undef SDDMMCSR_DECLARE_ENTRY

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}  // namespace runtime
namespace aten {
namespace {

using Launch = uint32_t (*)(uint32_t, aclrtStream, void*, void*, void*, void*, void*, void*, void*, void*, void*);

using LaunchTable = Launch[2][2][36];

// The type/index table layout is identical for the add and generic families.
#define SDDMMCSR_LAUNCH_TABLE(F32I32, F32I64, F64I32, F64I64) \
  {{{SDDMMCSR_FOR_EACH_KEY(F32I32)}, \
    {SDDMMCSR_FOR_EACH_KEY(F32I64)}}, \
   {{SDDMMCSR_FOR_EACH_KEY(F64I32)}, \
    {SDDMMCSR_FOR_EACH_KEY(F64I64)}}}
#define SDDMMCSR_ADD_F32_I32(Key, Lhs, Rhs, Bcast, Map) aclrtlaunch_sddmm_csr_add_f32_i32_##Key,
#define SDDMMCSR_ADD_F32_I64(Key, Lhs, Rhs, Bcast, Map) aclrtlaunch_sddmm_csr_add_f32_i64_##Key,
#define SDDMMCSR_ADD_F64_I32(Key, Lhs, Rhs, Bcast, Map) aclrtlaunch_sddmm_csr_add_f64_i32_##Key,
#define SDDMMCSR_ADD_F64_I64(Key, Lhs, Rhs, Bcast, Map) aclrtlaunch_sddmm_csr_add_f64_i64_##Key,
#define SDDMMCSR_F32_I32(Key, Lhs, Rhs, Bcast, Map) aclrtlaunch_sddmm_csr_f32_i32_##Key,
#define SDDMMCSR_F32_I64(Key, Lhs, Rhs, Bcast, Map) aclrtlaunch_sddmm_csr_f32_i64_##Key,
#define SDDMMCSR_F64_I32(Key, Lhs, Rhs, Bcast, Map) aclrtlaunch_sddmm_csr_f64_i32_##Key,
#define SDDMMCSR_F64_I64(Key, Lhs, Rhs, Bcast, Map) aclrtlaunch_sddmm_csr_f64_i64_##Key,
const LaunchTable AddLaunchers = SDDMMCSR_LAUNCH_TABLE(
    SDDMMCSR_ADD_F32_I32, SDDMMCSR_ADD_F32_I64,
    SDDMMCSR_ADD_F64_I32, SDDMMCSR_ADD_F64_I64);
const LaunchTable Launchers = SDDMMCSR_LAUNCH_TABLE(
    SDDMMCSR_F32_I32, SDDMMCSR_F32_I64, SDDMMCSR_F64_I32, SDDMMCSR_F64_I64);
#undef SDDMMCSR_ADD_F32_I32
#undef SDDMMCSR_ADD_F32_I64
#undef SDDMMCSR_ADD_F64_I32
#undef SDDMMCSR_ADD_F64_I64
#undef SDDMMCSR_F32_I32
#undef SDDMMCSR_F32_I64
#undef SDDMMCSR_F64_I32
#undef SDDMMCSR_F64_I64
#undef SDDMMCSR_LAUNCH_TABLE

void CheckAcl(aclError status) {
  CHECK_EQ(status, ACL_SUCCESS) << "SDDMMCsr failed, aclError=" << status;
}

uint64_t Align32(uint64_t bytes) { return (bytes + 31) / 32 * 32; }

void CheckArray(NDArray array, DGLContext ctx, DGLDataType dtype) {
  CHECK(array.defined());
  CHECK_EQ(array->ctx.device_type, ctx.device_type);
  CHECK_EQ(array->ctx.device_id, ctx.device_id);
  CHECK_EQ(array->dtype.code, dtype.code);
  CHECK_EQ(array->dtype.bits, dtype.bits);
  CHECK_EQ(array->dtype.lanes, 1);
  CHECK(array.IsContiguous()) << "SDDMMCsr expects contiguous arrays";
}

// Share UB accounting and feature/edge tile searches across CSR operations.
// The 8 KiB reserve covers alignment, stack, and the lhs buffer's bank offset.
SDDMMCsrTilingData MakeTiling(
    SDDMMCsrOp operation, uint32_t reduce_size, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target) {
  SDDMMCsrTilingData tiling{};
  tiling.numRows = csr.num_rows;
  tiling.numEdges = csr.indices->shape[0];
  tiling.lhsFeatureSize = bcast.lhs_len;
  tiling.rhsFeatureSize = bcast.rhs_len;
  tiling.outFeatureSize = bcast.out_len;
  tiling.lhsTarget = lhs_target;
  tiling.rhsTarget = rhs_target;
  tiling.hasEdgeMap = !IsNullArray(csr.data);
  tiling.hasLhsOffset = !bcast.lhs_offset.empty();
  tiling.hasRhsOffset = !bcast.rhs_offset.empty();
  tiling.useBcast = bcast.use_bcast;
  tiling.semanticVersion = SDDMMCsrSemanticVersion;
  tiling.operation = static_cast<uint32_t>(operation);
  tiling.reduceSize = reduce_size;

  int64_t core_count = 0;
  CheckAcl(aclrtGetDeviceInfo(out->ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &core_count));
  CHECK_GT(core_count, 0);
  tiling.blockNum = std::min<uint64_t>(core_count, tiling.numEdges);

  int64_t device_ub_bytes = 0, device_l2_bytes = 0;
  const auto ub_status = aclrtGetDeviceInfo(
      out->ctx.device_id, static_cast<aclrtDevAttr>(204), &device_ub_bytes);
  const uint64_t ub_bytes = ub_status == ACL_SUCCESS && device_ub_bytes > 0
                               ? device_ub_bytes : 192U * 1024U;
  const uint64_t element_bytes = out->dtype.bits / 8;
  const uint64_t id_bytes = csr.indptr->dtype.bits / 8;
  const uint32_t source_count = (lhs_target == 0) + (rhs_target == 0);
  const bool has_source = source_count != 0;
  const bool has_destination = lhs_target == 2 || rhs_target == 2;
  const uint32_t index_count = has_destination + tiling.hasEdgeMap;
  const uint64_t edges_per_core =
      (tiling.numEdges + tiling.blockNum - 1) / tiling.blockNum;
  tiling.indexCacheEdges = index_count == 0 ? 0 : std::max<uint64_t>(
      1, std::min<uint64_t>(edges_per_core, 64U * 1024U / (index_count * id_bytes)));
  // Dot writes small output rows; spend more UB on feature tiles than on
  // keeping an entire core's indices resident. Refills still bound each tile.
  if (operation == SDDMMCsrOp::Dot && element_bytes == 4 && !bcast.use_bcast)
    tiling.indexCacheEdges = std::min<uint32_t>(tiling.indexCacheEdges, 4096);
  const uint64_t index_bytes =
      index_count * Align32(tiling.indexCacheEdges * id_bytes) +
      (has_source ? 256U * id_bytes : 0);
  const uint32_t source_rows = has_destination
      ? SDDMMCsrSourceRows : SDDMMCsrContiguousSourceRows;
  const uint64_t max_input = std::max(bcast.lhs_len, bcast.rhs_len);
  const auto required_bytes = [&](uint64_t features, uint64_t edges) {
    const uint64_t buffered_rows = has_source
        ? 2 * (source_count * source_rows + edges) : 4 * edges;
    uint64_t bytes = 8U * 1024U + index_bytes +
                     buffered_rows * Align32(features * element_bytes);
    if (bcast.use_bcast) {
      const uint64_t temp_rows =
          std::max<uint64_t>(has_source ? source_rows : 0, edges);
      bytes += temp_rows * Align32(std::min(max_input, features) * element_bytes) +
               2 * Align32(features * sizeof(int64_t));
    }
    if (operation == SDDMMCsrOp::Dot) {
      const uint64_t groups = (features + reduce_size - 1) / reduce_size + 1;
      const uint64_t temp_row_bytes = element_bytes == 4 && reduce_size == 16
          ? Align32(features / 8 * element_bytes) : 32;
      bytes += 2 * edges * Align32(groups * element_bytes) + edges * temp_row_bytes;
    }
    return bytes;
  };
  // Search in 32-byte blocks so every full feature tile is already aligned.
  // A short row still reserves one full block; GM copies use its real length.
  const uint64_t elements_per_block = 32 / element_bytes;
  CHECK_LE(required_bytes(elements_per_block, 1), ub_bytes)
      << "Insufficient UB for SDDMMCsr";
  uint64_t low = 1;
  uint64_t high = (bcast.out_len + elements_per_block - 1) / elements_per_block;
  while (low < high) {
    const uint64_t mid = (low + high + 1) / 2;
    if (required_bytes(mid * elements_per_block, 1) <= ub_bytes) low = mid;
    else high = mid - 1;
  }
  tiling.featureTileSize = low * elements_per_block;
  const uint64_t tail_features = bcast.out_len % tiling.featureTileSize;
  tiling.tailFeatureStride = tail_features == 0 ? tiling.featureTileSize : Align32(tail_features * element_bytes) / element_bytes;
  low = 1;
  high = index_count == 0 ? 248 : std::min<uint32_t>(248, tiling.indexCacheEdges);
  while (low < high) {
    const uint64_t mid = (low + high + 1) / 2;
    if (required_bytes(tiling.featureTileSize, mid) <= ub_bytes) low = mid;
    else high = mid - 1;
  }
  tiling.tileEdges = low;
  tiling.tempFeatureSize = std::min<uint64_t>(max_input, tiling.featureTileSize);

  const auto l2_status = aclrtGetDeviceInfo(
      out->ctx.device_id, static_cast<aclrtDevAttr>(302), &device_l2_bytes);
  const uint64_t working_set = lhs.GetSize() + rhs.GetSize() + out.GetSize();
  if (l2_status == ACL_SUCCESS && device_l2_bytes > 0 &&
      working_set > static_cast<uint64_t>(device_l2_bytes) / 4 * 3)
    tiling.kernelFlags = SDDMMCsrStreamingOutputFlag;
  return tiling;
}

SDDMMCsrOp ParseOperation(const std::string& op) {
  if (op == "add") return SDDMMCsrOp::Add;
  if (op == "sub") return SDDMMCsrOp::Sub;
  if (op == "mul") return SDDMMCsrOp::Mul;
  if (op == "div") return SDDMMCsrOp::Div;
  if (op == "copy_lhs") return SDDMMCsrOp::CopyLhs;
  if (op == "copy_rhs") return SDDMMCsrOp::CopyRhs;
  CHECK_EQ(op, "dot") << "Unsupported Ascend CSR SDDMM operation: " << op;
  return SDDMMCsrOp::Dot;
}

// The shared gather path works in element coordinates, including dot products.
BcastOff PrepareBroadcast(SDDMMCsrOp operation, const BcastOff& bcast) {
  BcastOff result = bcast;
  if (operation == SDDMMCsrOp::CopyLhs || operation == SDDMMCsrOp::CopyRhs) {
    result.lhs_len = result.rhs_len = result.out_len;
    result.use_bcast = false;
    result.lhs_offset.clear();
    result.rhs_offset.clear();
  } else if (operation == SDDMMCsrOp::Dot) {
    CHECK_GT(bcast.reduce_size, 0);
    CHECK_LE(bcast.out_len, std::numeric_limits<uint32_t>::max() / bcast.reduce_size);
    result.out_len *= bcast.reduce_size;
    const auto expand = [&](const std::vector<int64_t>& offsets) {
      std::vector<int64_t> expanded;
      if (!offsets.empty()) {
        expanded.reserve(result.out_len);
        for (int64_t offset : offsets)
          for (int64_t k = 0; k < bcast.reduce_size; ++k)
            expanded.push_back(offset * bcast.reduce_size + k);
      }
      return expanded;
    };
    result.lhs_offset = expand(bcast.lhs_offset);
    result.rhs_offset = expand(bcast.rhs_offset);
  }
  return result;
}

}  // namespace

void SDDMMCsrAscend(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  const auto operation = ParseOperation(op);
  const auto ctx = out->ctx;
  const int64_t num_edges = csr.indices->shape[0];
  const bool has_map = !IsNullArray(csr.data);
  if (num_edges == 0 || bcast.out_len == 0) return;

  // The unused copy operand can be a null NDArray; never inspect its metadata.
  if (operation == SDDMMCsrOp::CopyLhs) {
    rhs = lhs;
    rhs_target = lhs_target;
  } else if (operation == SDDMMCsrOp::CopyRhs) {
    lhs = rhs;
    lhs_target = rhs_target;
  }
  CHECK_GE(lhs_target, 0);
  CHECK_LT(lhs_target, 3);
  CHECK_GE(rhs_target, 0);
  CHECK_LT(rhs_target, 3);
  CHECK_EQ(ctx.device_type, kDGLAscend);
  CHECK_EQ(out->dtype.code, kDGLFloat);
  CHECK(out->dtype.bits == 32 || out->dtype.bits == 64);
  CHECK(csr.indptr->dtype.bits == 32 || csr.indptr->dtype.bits == 64);
  CHECK_EQ(csr.indptr->dtype.code, kDGLInt);
  CheckArray(lhs, ctx, out->dtype);
  CheckArray(rhs, ctx, out->dtype);
  CheckArray(out, ctx, out->dtype);
  CheckArray(csr.indptr, ctx, csr.indptr->dtype);
  CheckArray(csr.indices, ctx, csr.indptr->dtype);
  if (has_map) CheckArray(csr.data, ctx, csr.indptr->dtype);
  const int64_t target_rows[] = {csr.num_rows, num_edges, csr.num_cols};
  CHECK_EQ(lhs->shape[0], target_rows[lhs_target]);
  CHECK_EQ(rhs->shape[0], target_rows[rhs_target]);
  CHECK_EQ(out->shape[0], num_edges);
  const auto features = PrepareBroadcast(operation, bcast);
  for (int64_t size : {features.lhs_len, features.rhs_len, features.out_len}) {
    CHECK_GT(size, 0);
    CHECK_LE(size, std::numeric_limits<uint32_t>::max());
  }

  CheckAcl(aclrtSetDevice(ctx.device_id));
#ifdef DGL_USE_PYTORCH_NPU_STREAM
  // Drain the producer's torch_npu task queue before the native ACL launch.
  // Keep this ordering in the Ascend host, independent of Python FFI mode.
  c10_npu::getCurrentNPUStream(
      static_cast<c10::DeviceIndex>(ctx.device_id)).synchronize();
#endif
  const auto stream = runtime::getCurrentAscendStream();
  const auto tiling = MakeTiling(
      operation, bcast.reduce_size, features, csr, lhs, rhs, out,
      lhs_target, rhs_target);
  const size_t lhs_bytes = features.lhs_offset.size() * sizeof(int64_t);
  const size_t rhs_bytes = features.rhs_offset.size() * sizeof(int64_t);
  runtime::Workspace<void> workspace(
      runtime::DeviceAPI::Get(ctx), ctx,
      Align32(sizeof(tiling)) + Align32(lhs_bytes) + rhs_bytes);
  auto* tiling_dev = static_cast<uint8_t*>(workspace.get());
  void* lhs_offset = lhs_bytes ? tiling_dev + Align32(sizeof(tiling)) : nullptr;
  void* rhs_offset = rhs_bytes
      ? tiling_dev + Align32(sizeof(tiling)) + Align32(lhs_bytes) : nullptr;
  CheckAcl(aclrtMemcpy(tiling_dev, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE));
  if (lhs_bytes) CheckAcl(aclrtMemcpy(lhs_offset, lhs_bytes, features.lhs_offset.data(), lhs_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  if (rhs_bytes) CheckAcl(aclrtMemcpy(rhs_offset, rhs_bytes, features.rhs_offset.data(), rhs_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  const uint32_t key = ((lhs_target * 3U + rhs_target) * 2U + static_cast<uint32_t>(features.use_bcast)) * 2U + static_cast<uint32_t>(has_map);
  const auto& launchers = operation == SDDMMCsrOp::Add ? AddLaunchers : Launchers;
  const auto launch = launchers[out->dtype.bits == 64][csr.indptr->dtype.bits == 64][key];
  CheckAcl(launch(tiling.blockNum, stream, csr.indptr->data, csr.indices->data,
      lhs->data, rhs->data, has_map ? csr.data->data : nullptr,
      lhs_offset, rhs_offset, out->data, tiling_dev));
  CheckAcl(aclrtSynchronizeStream(stream));
}

#define SDDMMCSR_DEFINE_HOST(Name, Op) \
  void SDDMMCsr##Name##Ascend( \
      const BcastOff& bcast, const CSRMatrix& csr, NDArray lhs, NDArray rhs, \
      NDArray out, int lhs_target, int rhs_target) { \
    SDDMMCsrAscend(Op, bcast, csr, lhs, rhs, out, lhs_target, rhs_target); \
  }
SDDMMCSR_DEFINE_HOST(Add, "add")
SDDMMCSR_DEFINE_HOST(Sub, "sub")
SDDMMCSR_DEFINE_HOST(Mul, "mul")
SDDMMCSR_DEFINE_HOST(Div, "div")
SDDMMCSR_DEFINE_HOST(CopyLhs, "copy_lhs")
SDDMMCSR_DEFINE_HOST(CopyRhs, "copy_rhs")
SDDMMCSR_DEFINE_HOST(Dot, "dot")
#undef SDDMMCSR_DEFINE_HOST

}  // namespace aten
}  // namespace dgl
#endif  // DGL_USE_ASCEND
