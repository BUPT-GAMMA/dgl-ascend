#include "kernel_operator.h"

using namespace AscendC;

// Tiling buffer layout (16 bytes):
//   [0..4)  uint32_t n   (element count)
//   [4..8)  padding
//   [8..16) val          (scalar fill value, typed per kernel)
//
// 混合方案（受 Duplicate API dtype 限制，见 ascendc-api-best-practices）：
//  - int32 / float32 : B档向量化（Duplicate 广播 + DataCopyPad 块搬出）
//  - int64 / double  : A档标量直写（DAV_2201 的 Duplicate 不支持 64-bit 类型）
//
// 保持 block_dim=1（对齐 DGL-Ascend 既有约定；核数须运行时 GetCoreNumAiv()
// 查询，禁止硬编码）。

namespace {
constexpr uint32_t FULL_TILE_LENGTH = 8192;
constexpr uint32_t FULL_DOUBLE_BUFFER = 2;

// B档：向量化路径（仅用于 Duplicate 支持的类型：int32/uint32/float 等 32-bit）
template <typename T>
__aicore__ inline void FullVectorKernel(GM_ADDR dst, uint32_t n, T val) {
  TPipe pipe;
  TQue<TPosition::VECOUT, FULL_DOUBLE_BUFFER> outQueue;
  pipe.InitBuffer(outQueue, FULL_DOUBLE_BUFFER, FULL_TILE_LENGTH * sizeof(T));

  GlobalTensor<T> dstGm;
  dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst), n);

  const uint32_t fullTiles = n / FULL_TILE_LENGTH;
  const uint32_t tailLen = n - fullTiles * FULL_TILE_LENGTH;

  for (uint32_t i = 0; i < fullTiles; i++) {
    LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();
    Duplicate<T>(outLocal, val, FULL_TILE_LENGTH);
    outQueue.template EnQue<T>(outLocal);
    LocalTensor<T> deq = outQueue.template DeQue<T>();
    DataCopyPad(dstGm[i * FULL_TILE_LENGTH], deq,
                {1, static_cast<uint16_t>(FULL_TILE_LENGTH * sizeof(T)), 0, 0});
    outQueue.FreeTensor(deq);
  }
  if (tailLen > 0) {
    LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();
    Duplicate<T>(outLocal, val, tailLen);
    outQueue.template EnQue<T>(outLocal);
    LocalTensor<T> deq = outQueue.template DeQue<T>();
    DataCopyPad(dstGm[fullTiles * FULL_TILE_LENGTH], deq,
                {1, static_cast<uint16_t>(tailLen * sizeof(T)), 0, 0});
    outQueue.FreeTensor(deq);
  }
}

// A档：标量直写路径（用于 64-bit 类型，Duplicate 不支持）
template <typename T>
__aicore__ inline void FullScalarKernel(GM_ADDR dst, uint32_t n, T val) {
  GlobalTensor<T> dstGm;
  dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst), n);
  for (uint32_t i = 0; i < n; i++) {
    dstGm.SetValue(i, val);
  }
}

__aicore__ inline uint32_t ReadNFromTiling(GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> nGm;
  nGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tiling_ptr), 1);
  return nGm.GetValue(0);
}
}  // namespace

// int32: B档向量化
extern "C" __global__ __aicore__ void full_i32(GM_ADDR dst, GM_ADDR tiling_ptr) {
  uint32_t n = ReadNFromTiling(tiling_ptr);
  GlobalTensor<int32_t> valGm;
  valGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ int32_t*>(
          reinterpret_cast<__gm__ char*>(tiling_ptr) + 8),
      1);
  int32_t val = valGm.GetValue(0);
  FullVectorKernel<int32_t>(dst, n, val);
}

// int64: A档标量（Duplicate 不支持 int64）
extern "C" __global__ __aicore__ void full_i64(GM_ADDR dst, GM_ADDR tiling_ptr) {
  uint32_t n = ReadNFromTiling(tiling_ptr);
  GlobalTensor<int64_t> valGm;
  valGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ int64_t*>(
          reinterpret_cast<__gm__ char*>(tiling_ptr) + 8),
      1);
  int64_t val = valGm.GetValue(0);
  FullScalarKernel<int64_t>(dst, n, val);
}

// float32: B档向量化
extern "C" __global__ __aicore__ void full_f32(GM_ADDR dst, GM_ADDR tiling_ptr) {
  uint32_t n = ReadNFromTiling(tiling_ptr);
  GlobalTensor<float> valGm;
  valGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ float*>(
          reinterpret_cast<__gm__ char*>(tiling_ptr) + 8),
      1);
  float val = valGm.GetValue(0);
  FullVectorKernel<float>(dst, n, val);
}

// double: A档标量（Duplicate 不支持 double）
extern "C" __global__ __aicore__ void full_f64(GM_ADDR dst, GM_ADDR tiling_ptr) {
  uint32_t n = ReadNFromTiling(tiling_ptr);
  GlobalTensor<double> valGm;
  valGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ double*>(
          reinterpret_cast<__gm__ char*>(tiling_ptr) + 8),
      1);
  double val = valGm.GetValue(0);
  FullScalarKernel<double>(dst, n, val);
}
