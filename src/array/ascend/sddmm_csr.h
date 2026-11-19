#ifndef DGL_ARRAY_ASCEND_SDDMM_CSR_H_
#define DGL_ARRAY_ASCEND_SDDMM_CSR_H_

#include <cstdint>

// Shared host/device tiling layout and semantic dispatch keys.
constexpr uint32_t SDDMMCsrSourceRows = 12;
constexpr uint32_t SDDMMCsrContiguousSourceRows = 32;
constexpr uint32_t SDDMMCsrStreamingOutputFlag = 2U;
constexpr uint32_t SDDMMCsrSemanticVersion = 2;

enum class SDDMMCsrOp : uint32_t {
    Add, Sub, Mul, Div, CopyLhs, CopyRhs, Dot
};

struct SDDMMCsrTilingData {
    uint64_t numRows;
    uint64_t numEdges;
    uint32_t lhsFeatureSize;
    uint32_t rhsFeatureSize;
    uint32_t outFeatureSize;
    uint32_t tileEdges;
    uint32_t lhsTarget;
    uint32_t rhsTarget;
    uint32_t blockNum;
    uint32_t hasEdgeMap;
    uint32_t hasLhsOffset;
    uint32_t hasRhsOffset;
    uint32_t useBcast;
    uint32_t kernelFlags;
    uint32_t featureTileSize;     // CPU 按 32B 对齐的特征 tile 容量，单位为元素。
    uint32_t tailFeatureStride;   // 最后一个特征 tile 的 32B 对齐 UB 行步长。
    uint32_t indexCacheEdges;     // indices/edge_map 缓存一次最多覆盖的边数。
    uint32_t tempFeatureSize;  // 广播临时缓冲区每行的最大输入特征数。
    uint32_t semanticVersion;
    uint32_t operation;
    uint32_t reduceSize;          // dot 沿最后一维归约，其余算子为 1。
};

// 根据不同参数 (key, lhsTarget, rhsTarget, useBroadcast, hasEdgeMap) 选择不同的Kernel
#define SDDMMCSR_FOR_EACH_KEY(APPLY) \
    APPLY(0, 0, 0, false, false) \
    APPLY(1, 0, 0, false, true) \
    APPLY(2, 0, 0, true, false) \
    APPLY(3, 0, 0, true, true) \
    APPLY(4, 0, 1, false, false) \
    APPLY(5, 0, 1, false, true) \
    APPLY(6, 0, 1, true, false) \
    APPLY(7, 0, 1, true, true) \
    APPLY(8, 0, 2, false, false) \
    APPLY(9, 0, 2, false, true) \
    APPLY(10, 0, 2, true, false) \
    APPLY(11, 0, 2, true, true) \
    APPLY(12, 1, 0, false, false) \
    APPLY(13, 1, 0, false, true) \
    APPLY(14, 1, 0, true, false) \
    APPLY(15, 1, 0, true, true) \
    APPLY(16, 1, 1, false, false) \
    APPLY(17, 1, 1, false, true) \
    APPLY(18, 1, 1, true, false) \
    APPLY(19, 1, 1, true, true) \
    APPLY(20, 1, 2, false, false) \
    APPLY(21, 1, 2, false, true) \
    APPLY(22, 1, 2, true, false) \
    APPLY(23, 1, 2, true, true) \
    APPLY(24, 2, 0, false, false) \
    APPLY(25, 2, 0, false, true) \
    APPLY(26, 2, 0, true, false) \
    APPLY(27, 2, 0, true, true) \
    APPLY(28, 2, 1, false, false) \
    APPLY(29, 2, 1, false, true) \
    APPLY(30, 2, 1, true, false) \
    APPLY(31, 2, 1, true, true) \
    APPLY(32, 2, 2, false, false) \
    APPLY(33, 2, 2, false, true) \
    APPLY(34, 2, 2, true, false) \
    APPLY(35, 2, 2, true, true)

// The device translation unit only needs the shared definitions above.
#ifndef DGL_SDDMM_CSR_DEVICE
#include <dgl/array.h>
#include <dgl/bcast.h>

namespace dgl {
namespace aten {

void SDDMMCsrAscend(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);

#define SDDMMCSR_DECLARE_HOST(Name) \
  void SDDMMCsr##Name##Ascend( \
      const BcastOff& bcast, const CSRMatrix& csr, NDArray lhs, NDArray rhs, \
      NDArray out, int lhs_target, int rhs_target);
SDDMMCSR_DECLARE_HOST(Add)
SDDMMCSR_DECLARE_HOST(Sub)
SDDMMCSR_DECLARE_HOST(Mul)
SDDMMCSR_DECLARE_HOST(Div)
SDDMMCSR_DECLARE_HOST(CopyLhs)
SDDMMCSR_DECLARE_HOST(CopyRhs)
SDDMMCSR_DECLARE_HOST(Dot)
#undef SDDMMCSR_DECLARE_HOST

}  // namespace aten
}  // namespace dgl
#endif  // DGL_SDDMM_CSR_DEVICE

#endif  // DGL_ARRAY_ASCEND_SDDMM_CSR_H_
