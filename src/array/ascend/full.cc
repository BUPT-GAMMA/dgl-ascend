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

extern "C" uint32_t aclrtlaunch_full_i32(
    uint32_t blockDim, aclrtStream stream, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_full_i64(
    uint32_t blockDim, aclrtStream stream, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_full_f32(
    uint32_t blockDim, aclrtStream stream, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_full_f64(
    uint32_t blockDim, aclrtStream stream, void* dst, void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}
}
#endif

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>

#include "../array_op.h"

namespace dgl {
namespace aten {
namespace impl {

#ifdef DGL_USE_ASCEND

namespace {
// Tiling buffer layout (16 bytes): [0..4) uint32 n; [4..8) pad; [8..16) val.
constexpr size_t kFullTilingBytes = 16;

template <typename CppType, typename LaunchFn>
NDArray FullAscendImpl(CppType val, int64_t length, DGLContext ctx,
                       LaunchFn launch) {
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  NDArray ret =
      NDArray::Empty({length}, DGLDataTypeTraits<CppType>::dtype, ctx);
  if (length == 0) return ret;
  char host_tiling[kFullTilingBytes] = {0};
  *reinterpret_cast<uint32_t*>(host_tiling) =
      static_cast<uint32_t>(length);
  *reinterpret_cast<CppType*>(host_tiling + 8) = val;
  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev, kFullTilingBytes,
                           ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tiling_dev, kFullTilingBytes, host_tiling,
                           kFullTilingBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  auto stream = dgl::runtime::getCurrentAscendStream();
  uint32_t block_dim = 1;
  launch(block_dim, stream, ret->data, tiling_dev);
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  if (tiling_dev) ASCEND_CALL(aclrtFree(tiling_dev));
  return ret;
}
}  // namespace

#define ASCEND_FULL_SPEC(CppType, launch_name)                                  \
  template <>                                                                    \
  NDArray Full<kDGLAscend, CppType>(CppType val, int64_t length,                \
                                    DGLContext ctx) {                           \
    return FullAscendImpl<CppType>(                                              \
        val, length, ctx,                                                       \
        [](uint32_t blockDim, aclrtStream stream, void* dst, void* tiling) {     \
          launch_name(blockDim, stream, dst, tiling);                           \
        });                                                                      \
  }

ASCEND_FULL_SPEC(int32_t, aclrtlaunch_full_i32)
ASCEND_FULL_SPEC(int64_t, aclrtlaunch_full_i64)
ASCEND_FULL_SPEC(float, aclrtlaunch_full_f32)
ASCEND_FULL_SPEC(double, aclrtlaunch_full_f64)

#else  // DGL_USE_ASCEND

template <>
NDArray Full<kDGLAscend, int32_t>(int32_t, int64_t, DGLContext) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}
template <>
NDArray Full<kDGLAscend, int64_t>(int64_t, int64_t, DGLContext) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}
template <>
NDArray Full<kDGLAscend, float>(float, int64_t, DGLContext) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}
template <>
NDArray Full<kDGLAscend, double>(double, int64_t, DGLContext) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}

#endif  // DGL_USE_ASCEND

}  // namespace impl
}  // namespace aten
}  // namespace dgl
