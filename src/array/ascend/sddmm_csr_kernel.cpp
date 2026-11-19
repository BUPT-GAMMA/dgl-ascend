#include "kernel_operator.h"

#define DGL_SDDMM_CSR_DEVICE
#include "sddmm_csr.h"
#undef DGL_SDDMM_CSR_DEVICE

// Ascend 910_93 moves FP64 as uint64_t. All arithmetic below uses integer
// instructions and rounds once to nearest, ties to even.
constexpr uint64_t kFp64FractionMask = (1ULL << 52) - 1;
constexpr uint64_t kFp64HiddenBit = 1ULL << 52;
constexpr uint64_t kFp64QuietBit = 1ULL << 51;

__aicore__ inline uint64_t ShiftRightJam(uint64_t value, uint32_t distance)
{
    if (distance == 0) {
        return value;
    }
    if (distance < 64) {
        const uint64_t lostMask = (1ULL << distance) - 1;
        return (value >> distance) | ((value & lostMask) != 0 ? 1ULL : 0ULL);
    }
    return value != 0 ? 1ULL : 0ULL;
}


// 拆解 Double 数据类型进行相加
__aicore__ inline uint64_t AddFp64Bits(uint64_t lhsBits, uint64_t rhsBits)
{
    constexpr uint64_t magnitudeMask = 0x7fffffffffffffffULL;
    constexpr uint64_t infinityBits = 0x7ff0000000000000ULL;

    // 排序前处理 NaN，保留原始左操作数的优先级。
    if ((lhsBits & magnitudeMask) > infinityBits) return lhsBits | kFp64QuietBit;
    if ((rhsBits & magnitudeMask) > infinityBits) return rhsBits | kFp64QuietBit;

    // 去掉符号位后，位模式的大小顺序与绝对值一致。
    if ((lhsBits & magnitudeMask) < (rhsBits & magnitudeMask)) {
        const uint64_t savedBits = lhsBits;
        lhsBits = rhsBits;
        rhsBits = savedBits;
    }
    if ((lhsBits & magnitudeMask) == infinityBits) {
        if ((rhsBits & magnitudeMask) == infinityBits && (lhsBits >> 63) != (rhsBits >> 63)) {
            return 0x7ff8000000000000ULL;
        }
        return lhsBits;
    }
    if ((rhsBits & magnitudeMask) == 0) {
        // 两个零相加时，仅 -0 + -0 得到 -0。
        return (lhsBits & magnitudeMask) == 0 ? (lhsBits & rhsBits) : lhsBits;
    }

    const uint32_t lhsSign = static_cast<uint32_t>(lhsBits >> 63);
    const uint32_t rhsSign = static_cast<uint32_t>(rhsBits >> 63);
    uint32_t lhsExponent = static_cast<uint32_t>((lhsBits >> 52) & 0x7ffULL);
    uint32_t rhsExponent = static_cast<uint32_t>((rhsBits >> 52) & 0x7ffULL);
    uint64_t lhsSignificand = (lhsBits & kFp64FractionMask) | (lhsExponent == 0 ? 0ULL : kFp64HiddenBit);
    uint64_t rhsSignificand = (rhsBits & kFp64FractionMask) | (rhsExponent == 0 ? 0ULL : kFp64HiddenBit);
    // 非规格化数没有隐藏位，其有效指数与最小规格化数相同。
    lhsExponent = lhsExponent == 0 ? 1 : lhsExponent;
    rhsExponent = rhsExponent == 0 ? 1 : rhsExponent;

    // 低三位用于 guard、round、sticky 舍入。
    lhsSignificand <<= 3;
    rhsSignificand = ShiftRightJam(rhsSignificand << 3, lhsExponent - rhsExponent);
    uint64_t resultSignificand = 0;
    uint32_t resultExponent = lhsExponent;

    if (lhsSign == rhsSign) {
        resultSignificand = lhsSignificand + rhsSignificand;
        if ((resultSignificand & (1ULL << 56)) != 0) {
            resultSignificand = ShiftRightJam(resultSignificand, 1);
            ++resultExponent;
        }
    } else {
        resultSignificand = lhsSignificand - rhsSignificand;
        if (resultSignificand == 0) {
            return 0;
        }
        // 将最高有效位移到 bit 55，并在指数降到 1 时保留非规格化结果。
        const uint32_t leadingZeros = static_cast<uint32_t>(AscendC::CountLeadingZero(resultSignificand)) - 8;
        const uint32_t normalizeShift = leadingZeros < resultExponent ? leadingZeros : resultExponent - 1;
        resultSignificand <<= normalizeShift;
        resultExponent -= normalizeShift;
    }

    const uint32_t roundBits = static_cast<uint32_t>(resultSignificand & 7ULL);
    uint64_t roundedSignificand = resultSignificand >> 3;
    if (roundBits > 4U || (roundBits == 4U && (roundedSignificand & 1ULL) != 0)) {
        ++roundedSignificand;
        if ((roundedSignificand & (1ULL << 53)) != 0) {
            roundedSignificand >>= 1;
            ++resultExponent;
        }
    }

    if (resultExponent >= 0x7ffU) {
        return (static_cast<uint64_t>(lhsSign) << 63) | infinityBits;
    }
    const uint64_t storedExponent = resultExponent == 1 && roundedSignificand < kFp64HiddenBit ? 0ULL : resultExponent;
    return (static_cast<uint64_t>(lhsSign) << 63) | (storedExponent << 52) |
           (roundedSignificand & kFp64FractionMask);
}

__aicore__ inline uint64_t SubFp64Bits(uint64_t lhsBits, uint64_t rhsBits)
{
    // Negate finite values and infinities, preserving a NaN's original sign.
    if ((lhsBits & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL)
        return lhsBits | kFp64QuietBit;
    if ((rhsBits & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL)
        return rhsBits | kFp64QuietBit;
    return AddFp64Bits(lhsBits, rhsBits ^ (1ULL << 63));
}

// Return a normalized 53-bit significand and its signed, biased exponent.
// Callers exclude zero and exceptional operands before normalization.
__aicore__ inline uint64_t NormalizeFp64Bits(uint64_t bits, int32_t &exponent)
{
    exponent = static_cast<int32_t>((bits >> 52) & 0x7ffULL);
    uint64_t significand = bits & kFp64FractionMask;
    if (exponent != 0) return significand | kFp64HiddenBit;
    const uint32_t shift =
        static_cast<uint32_t>(AscendC::CountLeadingZero(significand)) - 11;
    exponent = 1 - static_cast<int32_t>(shift);
    return significand << shift;
}

// significand has its leading bit at bit 55, followed by guard/round/sticky.
// Delaying rounding until after subnormal alignment avoids double rounding.
__aicore__ inline uint64_t RoundFp64Bits(
    uint64_t sign, int32_t exponent, uint64_t significand)
{
    if (exponent <= 0) {
        significand = ShiftRightJam(significand, static_cast<uint32_t>(1 - exponent));
        exponent = 1;
    }
    const uint32_t roundBits = static_cast<uint32_t>(significand & 7ULL);
    uint64_t rounded = significand >> 3;
    if (roundBits > 4U || (roundBits == 4U && (rounded & 1ULL) != 0)) {
        ++rounded;
        if ((rounded & (1ULL << 53)) != 0) {
            rounded >>= 1;
            ++exponent;
        }
    }
    if (exponent >= 0x7ff) return sign | 0x7ff0000000000000ULL;
    const uint64_t storedExponent =
        exponent == 1 && rounded < kFp64HiddenBit ? 0ULL : static_cast<uint64_t>(exponent);
    return sign | (storedExponent << 52) | (rounded & kFp64FractionMask);
}

__aicore__ inline uint64_t MulFp64Bits(uint64_t lhsBits, uint64_t rhsBits)
{
    constexpr uint64_t magnitudeMask = 0x7fffffffffffffffULL;
    constexpr uint64_t infinityBits = 0x7ff0000000000000ULL;
    const uint64_t lhsMagnitude = lhsBits & magnitudeMask;
    const uint64_t rhsMagnitude = rhsBits & magnitudeMask;
    const uint64_t sign = (lhsBits ^ rhsBits) & (1ULL << 63);
    if (lhsMagnitude > infinityBits) return lhsBits | kFp64QuietBit;
    if (rhsMagnitude > infinityBits) return rhsBits | kFp64QuietBit;
    if (lhsMagnitude == infinityBits || rhsMagnitude == infinityBits) {
        if (lhsMagnitude == 0 || rhsMagnitude == 0) return infinityBits | kFp64QuietBit;
        return sign | infinityBits;
    }
    if (lhsMagnitude == 0 || rhsMagnitude == 0) return sign;

    int32_t lhsExponent, rhsExponent;
    const uint64_t lhsSignificand = NormalizeFp64Bits(lhsBits, lhsExponent);
    const uint64_t rhsSignificand = NormalizeFp64Bits(rhsBits, rhsExponent);
    constexpr uint64_t lowMask = 0xffffffffULL;
    const uint64_t product00 = (lhsSignificand & lowMask) * (rhsSignificand & lowMask);
    const uint64_t product01 = (lhsSignificand & lowMask) * (rhsSignificand >> 32);
    const uint64_t product10 = (lhsSignificand >> 32) * (rhsSignificand & lowMask);
    const uint64_t product11 = (lhsSignificand >> 32) * (rhsSignificand >> 32);
    const uint64_t middle =
        (product00 >> 32) + (product01 & lowMask) + (product10 & lowMask);
    const uint64_t low = (middle << 32) | (product00 & lowMask);
    const uint64_t high =
        product11 + (product01 >> 32) + (product10 >> 32) + (middle >> 32);
    // The exact 106-bit product has its leading bit at position 104 or 105.
    const uint32_t carry = static_cast<uint32_t>((high >> 41) & 1ULL);
    const uint32_t shift = 49 + carry;
    const uint64_t significand = (high << (64 - shift)) | (low >> shift) |
        ((low & ((1ULL << shift) - 1)) != 0 ? 1ULL : 0ULL);
    return RoundFp64Bits(
        sign, lhsExponent + rhsExponent - 1023 + static_cast<int32_t>(carry), significand);
}

__aicore__ inline uint64_t DivFp64Bits(uint64_t lhsBits, uint64_t rhsBits)
{
    constexpr uint64_t magnitudeMask = 0x7fffffffffffffffULL;
    constexpr uint64_t infinityBits = 0x7ff0000000000000ULL;
    const uint64_t lhsMagnitude = lhsBits & magnitudeMask;
    const uint64_t rhsMagnitude = rhsBits & magnitudeMask;
    const uint64_t sign = (lhsBits ^ rhsBits) & (1ULL << 63);
    if (lhsMagnitude > infinityBits) return lhsBits | kFp64QuietBit;
    if (rhsMagnitude > infinityBits) return rhsBits | kFp64QuietBit;
    if ((lhsMagnitude == 0 && rhsMagnitude == 0) ||
        (lhsMagnitude == infinityBits && rhsMagnitude == infinityBits))
        return infinityBits | kFp64QuietBit;
    if (lhsMagnitude == infinityBits || rhsMagnitude == 0) return sign | infinityBits;
    if (rhsMagnitude == infinityBits || lhsMagnitude == 0) return sign;

    int32_t lhsExponent, rhsExponent;
    uint64_t remainder = NormalizeFp64Bits(lhsBits, lhsExponent);
    const uint64_t divisor = NormalizeFp64Bits(rhsBits, rhsExponent);
    int32_t exponent = lhsExponent - rhsExponent + 1023;
    if (remainder < divisor) {
        remainder <<= 1;
        --exponent;
    }
    // Generate the normalized significand plus three rounding bits. A nonzero
    // remainder supplies the sticky bit; every intermediate fits in 54 bits.
    uint64_t quotient = 0;
    for (uint32_t bit = 0; bit < 56; ++bit) {
        quotient <<= 1;
        if (remainder >= divisor) {
            remainder -= divisor;
            quotient |= 1ULL;
        }
        remainder <<= 1;
    }
    quotient |= remainder != 0 ? 1ULL : 0ULL;
    return RoundFp64Bits(sign, exponent, quotient);
}

// CSR readers, feature broadcasting, computation, and output placement.
template <typename T, typename IdType, uint32_t InputLhsTarget, uint32_t InputRhsTarget, bool UseBroadcast, bool HasEdgeMap, bool AddOnly = true>
class SDDMMCsrKernel {
    static constexpr bool IsFp32 = sizeof(T) == sizeof(float);
    // InputLhs/InputRhs 对应原始输入；内部 lhs/rhs 将源节点操作数优先放到 lhs 槽。
    // 例如 v+u 内部按 u+v 执行，特征、target 和 offset 必须一起交换。
    static constexpr bool SwapInputsHs = InputRhsTarget == 0 && InputLhsTarget != 0;
    static constexpr uint32_t LhsTarget = SwapInputsHs ? InputRhsTarget : InputLhsTarget;
    static constexpr uint32_t RhsTarget = SwapInputsHs ? InputLhsTarget : InputRhsTarget;
    static constexpr bool HasSourceOperand = LhsTarget == 0;
    static constexpr bool BothOperandsUseSource = LhsTarget == 0 && RhsTarget == 0;
    static constexpr bool HasDestinationOperand = InputLhsTarget == 2 || InputRhsTarget == 2;
    static constexpr uint32_t MaxSourceRowsPerTile = HasDestinationOperand ? SDDMMCsrSourceRows : SDDMMCsrContiguousSourceRows;
    static constexpr uint32_t BufferCount = 2;
    static constexpr uint32_t IndptrCacheEntryCount = 256;  // 每次搬运 256 个行元素

    struct BroadcastPlan {
        uint64_t minInputOffset = 0;    // 当前特征块引用的最小输入元素下标。
        uint32_t inputSpan = 0;         // 最大下标 - 最小下标 + 1，单位为元素。
        uint32_t tempRowStride = 0;  // inputSpan 32字节对齐后的值，offsets 每行对齐后的跨度，单位为元素。
        bool isIdentity = true;         // 是否为恒等映射，输入下标与输出特征位置一致，可直接连续搬运。
        bool fitsTemp = true;        // inputSpan 是否落在临时广播缓冲区单行可容纳的特征元素数量（inputSpan <=tempFeatureCapacity_（广播 temp 每行的最大输入特征数））。
    };

public:
    __aicore__ inline void Run(AscendC::TPipe &pipe, GM_ADDR indptr, GM_ADDR indices, GM_ADDR inputLhs, GM_ADDR inputRhs, GM_ADDR edgeMap, GM_ADDR inputLhsOffset, GM_ADDR inputRhsOffset, GM_ADDR out, const SDDMMCsrTilingData &tiling)
    {
        if (tiling.numEdges == 0) return;
        numRows_ = tiling.numRows;
        edgeBegin_ = tiling.numEdges * AscendC::GetBlockIdx() / tiling.blockNum;
        edgeEnd_ = tiling.numEdges * (AscendC::GetBlockIdx() + 1) / tiling.blockNum;
        if (edgeBegin_ >= edgeEnd_) return;

        operation_ = AddOnly ? SDDMMCsrOp::Add : static_cast<SDDMMCsrOp>(tiling.operation);
        reduceSize_ = tiling.reduceSize;
        tileEdges_ = tiling.tileEdges;
        tileFeatureSize_ = tiling.featureTileSize;
        outFeatureSize_ = tiling.outFeatureSize;
        // A complete aligned FP32 tile never reads or writes scalar feature
        // values. Let MTE2 wait for slot reuse without stalling scalar issue.
        dotVectorOnly_ = IsFp32 && !UseBroadcast && IsDot() &&
            outFeatureSize_ <= tileFeatureSize_ && reduceSize_ % 8 == 0 &&
            AlignedFeature(outFeatureSize_) / 8 < 256 &&
            (!HasEdgeMap || (reduceSize_ == 16 &&
                (outFeatureSize_ / reduceSize_) % 8 == 0));
        lhsFeatureSize_ = SwapInputsHs ? tiling.rhsFeatureSize : tiling.lhsFeatureSize;
        rhsFeatureSize_ = SwapInputsHs ? tiling.lhsFeatureSize : tiling.rhsFeatureSize;
        hasLhsOffset_ = SwapInputsHs ? tiling.hasRhsOffset != 0 : tiling.hasLhsOffset != 0;
        hasRhsOffset_ = SwapInputsHs ? tiling.hasLhsOffset != 0 : tiling.hasRhsOffset != 0;
        indexCacheEdges_ = tiling.indexCacheEdges;
        tempFeatureCapacity_ = tiling.tempFeatureSize;  // 临时缓冲区单行可以容纳的特征元素
        lhsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(SwapInputsHs ? inputRhs : inputLhs));
        rhsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(SwapInputsHs ? inputLhs : inputRhs));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

        if ((tiling.kernelFlags & SDDMMCsrStreamingOutputFlag) != 0) outGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if constexpr (HasSourceOperand) indptrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType *>(indptr));
        if constexpr (HasDestinationOperand) indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType *>(indices));
        if constexpr (HasEdgeMap) edgeMapGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType *>(edgeMap));
        if constexpr (UseBroadcast) {
            if (hasLhsOffset_) lhsOffsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(SwapInputsHs ? inputRhsOffset : inputLhsOffset));
            if (hasRhsOffset_) rhsOffsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(SwapInputsHs ? inputLhsOffset : inputRhsOffset));
        }

        lhsSlotElements_ = (HasSourceOperand ? MaxSourceRowsPerTile : tileEdges_) * tileFeatureSize_;
        rhsSlotElements_ = (tileEdges_ + (BothOperandsUseSource ? MaxSourceRowsPerTile : 0)) * tileFeatureSize_;
        pipe.InitBuffer(lhsBuf_, BufferCount * lhsSlotElements_ * sizeof(T) + 32);
        pipe.InitBuffer(rhsBuf_, BufferCount * rhsSlotElements_ * sizeof(T));
        lhsLocal_ = lhsBuf_.Get<T>();
        rhsLocal_ = rhsBuf_.Get<T>();
        if constexpr (HasSourceOperand) {
            pipe.InitBuffer(indptrBuf_, IndptrCacheEntryCount * sizeof(IdType));
            indptrLocal_ = indptrBuf_.Get<IdType>();
        }
        if constexpr (HasDestinationOperand) {
            pipe.InitBuffer(indicesBuf_, Align32(indexCacheEdges_ * sizeof(IdType)));
            indicesLocal_ = indicesBuf_.Get<IdType>();
        }
        if constexpr (HasEdgeMap) {
            pipe.InitBuffer(edgeMapBuf_, Align32(indexCacheEdges_ * sizeof(IdType)));
            edgeMapLocal_ = edgeMapBuf_.Get<IdType>();
        }
        if constexpr (UseBroadcast) {
            const uint32_t tempRowCount = HasSourceOperand && MaxSourceRowsPerTile > tileEdges_ ? MaxSourceRowsPerTile : tileEdges_;
            pipe.InitBuffer(broadcastTempBuf_, tempRowCount * AlignedFeature(tempFeatureCapacity_) * sizeof(T));
            pipe.InitBuffer(lhsOffsetBuf_, tileFeatureSize_ * sizeof(int64_t));
            pipe.InitBuffer(rhsOffsetBuf_, tileFeatureSize_ * sizeof(int64_t));
        }

        if (IsDot()) {
            dotSlotElements_ = tileEdges_ * AlignedFeature((tileFeatureSize_ - 1) / reduceSize_ + 2);
            pipe.InitBuffer(dotOutBuf_, BufferCount * dotSlotElements_ * sizeof(T));
            const uint32_t tempRowBytes = IsFp32 && reduceSize_ == 16
                ? Align32(tileFeatureSize_ / 8 * sizeof(T)) : 32;
            pipe.InitBuffer(dotTempBuf_, tileEdges_ * tempRowBytes);
        }

        InitEvents(pipe);

        uint64_t initialSourceRow = 0;
        if constexpr (HasSourceOperand) {
            uint64_t low = 0, high = numRows_;
            while (low < high) {
                const uint64_t mid = (low + high + 1) / 2;
                if (static_cast<uint64_t>(indptrGm_.GetValue(mid)) <= edgeBegin_) low = mid;
                else high = mid - 1;
            }
            initialSourceRow = low;
        }

        for (uint64_t featureIndex = 0; featureIndex < outFeatureSize_; featureIndex += tileFeatureSize_) {
            featureBegin_ = static_cast<uint32_t>(featureIndex);
            // 每一个 Tiling 的每一行有 featureCount_ 个元素
            featureCount_ = outFeatureSize_ - featureBegin_ < tileFeatureSize_ ? outFeatureSize_ - featureBegin_ : tileFeatureSize_;
            localFeatureStride_ = featureCount_ == tileFeatureSize_ ? tileFeatureSize_ : tiling.tailFeatureStride;
            if constexpr (UseBroadcast) {
                // 对于每一个 Tile 准备一个 BroadcastPlan , 用来判断使用标量广播还是 Gather 广播，Gather 广播需要将 offsets 所涉及的源特征搬入 UB
                lhsBroadcastPlan_ = PrepareBroadcastPlan(lhsOffsetsGm_, lhsOffsetBuf_, hasLhsOffset_, lhsFeatureSize_);
                rhsBroadcastPlan_ = PrepareBroadcastPlan(rhsOffsetsGm_, rhsOffsetBuf_, hasRhsOffset_, rhsFeatureSize_);
            }
            // 无广播的源节点路径中，每个特征块只设置一次向量 mask，尾块也按实际宽度设置。
            if constexpr (IsFp32 && HasSourceOperand && !UseBroadcast) {
                AscendC::SetMaskNorm();
                AscendC::SetVectorMask<T>(static_cast<int32_t>(featureCount_ < 64 ? featureCount_ : 64));
            }
            for (uint32_t bufferIndex = 0; bufferIndex < BufferCount; ++bufferIndex)
                SetForBuffer(bufferIndex);
            uint64_t sourceRow = initialSourceRow;
            uint32_t tileIndex = 0;
            for (uint64_t csrEdge = edgeBegin_; csrEdge < edgeEnd_; ++tileIndex) {
                CopyIndices(csrEdge);
                uint32_t tileEdgeCount = static_cast<uint32_t>(Min(edgeEnd_ - csrEdge, tileEdges_));  // tileEdgeCount 不能超过当前核处理的边界
                if constexpr (HasDestinationOperand || HasEdgeMap) tileEdgeCount = static_cast<uint32_t>(Min(tileEdgeCount, edgeCacheEnd_ - csrEdge));  // tileEdgeCount 不能超过搬运边的边界
                uint64_t tileSourceRowBegin = sourceRow;
                uint32_t sourceRowCount = 0;
                // 统计每个源节点要计算多少条边
                uint32_t sourceSegmentEnds[MaxSourceRowsPerTile];
                if constexpr (HasSourceOperand) {
                    // 获得 csrEdge 的源节点
                    while (GetSourceRowEdgeEnd(tileSourceRowBegin) <= csrEdge) ++tileSourceRowBegin;
                    const uint64_t tileSourceRowEnd = Min(numRows_ - 1, tileSourceRowBegin + MaxSourceRowsPerTile - 1);
                    tileEdgeCount = static_cast<uint32_t>(Min(tileEdgeCount, GetSourceRowEdgeEnd(tileSourceRowEnd) - csrEdge));
                    uint64_t rowEnd;
                    do {
                        rowEnd = GetSourceRowEdgeEnd(tileSourceRowBegin + sourceRowCount);
                        sourceSegmentEnds[sourceRowCount++] = static_cast<uint32_t>(Min(rowEnd, csrEdge + tileEdgeCount) - csrEdge);
                    } while (rowEnd < csrEdge + tileEdgeCount);
                    sourceRow = tileSourceRowBegin + sourceRowCount - 1 + (rowEnd == csrEdge + tileEdgeCount ? 1 : 0);
                }
                const uint32_t bufferSlot = tileIndex % BufferCount;
                WaitForBuffer(bufferSlot);
                auto lhsTile = lhsLocal_[bufferSlot * lhsSlotElements_];
                auto rhsTile = rhsLocal_[bufferSlot * rhsSlotElements_];
                const uint32_t outputLocalOffset = BothOperandsUseSource ? MaxSourceRowsPerTile * localFeatureStride_ : 0;
                if (!IsCopy()) CopyOperand<LhsTarget>(lhsTile, lhsGm_, lhsFeatureSize_, lhsOffsetBuf_, lhsBroadcastPlan_, csrEdge, tileEdgeCount, tileSourceRowBegin, sourceRowCount);
                CopyOperand<RhsTarget>(rhsTile, rhsGm_, rhsFeatureSize_, rhsOffsetBuf_, rhsBroadcastPlan_, csrEdge, tileEdgeCount, tileSourceRowBegin, sourceRowCount);
                if constexpr (IsFp32) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVectorEvents_[bufferSlot]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVectorEvents_[bufferSlot]);
                    if constexpr (UseBroadcast) {
                        AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarToVectorInputsEvent_);
                        AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarToVectorInputsEvent_);
                    }
                } else {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarEvents_[bufferSlot]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarEvents_[bufferSlot]);
                }
                // 输出会直接放在 rhsTile 里面
                auto outTile = rhsTile[outputLocalOffset];
                if constexpr (HasSourceOperand) {
                    uint32_t segmentBegin = 0;
                    for (uint32_t sourceRowIndex = 0; sourceRowIndex < sourceRowCount; ++sourceRowIndex) {
                        const uint32_t segmentEnd = sourceSegmentEnds[sourceRowIndex];
                        if (segmentEnd != segmentBegin) {
                            auto lhsSegment = lhsTile[sourceRowIndex * localFeatureStride_];
                            auto rhsSegment = rhsTile[BothOperandsUseSource ? sourceRowIndex * localFeatureStride_ : segmentBegin * localFeatureStride_];
                            ComputeTile(outTile[segmentBegin * localFeatureStride_], lhsSegment, rhsSegment, segmentEnd - segmentBegin);
                        }
                        segmentBegin = segmentEnd;
                    }
                } else {
                    ComputeTile(outTile, lhsTile, rhsTile, tileEdgeCount);
                }
                auto resultTile = outTile;
                if (IsDot()) {
                    resultTile = dotOutBuf_.Get<T>()[bufferSlot * dotSlotElements_];
                    ReduceDot(resultTile, outTile, csrEdge, tileEdgeCount);
                }
                if constexpr (IsFp32) {
                    if (IsDot() && dotScalarOutput_) {
                        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(dotScalarToMte3Event_);
                        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(dotScalarToMte3Event_);
                    } else {
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3Events_[bufferSlot]);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3Events_[bufferSlot]);
                    }
                } else {
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3Events_[bufferSlot]);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3Events_[bufferSlot]);
                }
                if (IsDot()) CopyOutDot(resultTile, csrEdge, tileEdgeCount);
                else CopyOut(resultTile, csrEdge, tileEdgeCount);
                SetForBuffer(bufferSlot);
                csrEdge += tileEdgeCount;
            }
            for (uint32_t bufferIndex = 0; bufferIndex < BufferCount; ++bufferIndex)
                WaitForBuffer(bufferIndex);
            if (IsDot() && featureIndex + tileFeatureSize_ < outFeatureSize_) {
                // A reduction spanning two feature tiles reads its previous partial.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(dotMte3ToScalarEvent_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(dotMte3ToScalarEvent_);
            }
            if constexpr (IsFp32 && !UseBroadcast) {
                if (featureIndex + tileFeatureSize_ < outFeatureSize_) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarFeatureDrainEvent_);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarFeatureDrainEvent_);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        ReleaseEvents(pipe);
    }

private:
    __aicore__ inline uint32_t Align32(uint32_t byteCount) const { return (byteCount + 31U) & ~31U; }
    __aicore__ inline uint32_t AlignedFeature(uint32_t featureCount) const {
        return Align32(featureCount * sizeof(T)) / sizeof(T);
    }
    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const { return lhs < rhs ? lhs : rhs; }
    __aicore__ inline void InitEvents(AscendC::TPipe &pipe) {
        if (IsDot()) {
            dotMte3ToScalarEvent_ = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::MTE3_S>());
            dotMte2ToScalarEvent_ = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::MTE2_S>());
            if constexpr (IsFp32) {
                dotVectorToScalarEvent_ = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::V_S>());
                dotScalarToMte3Event_ = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::S_MTE3>());
            }
        }
        for (uint32_t slot = 0; slot < BufferCount; ++slot) {
            if constexpr (IsFp32) {
                mte2ToVectorEvents_[slot] = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::MTE2_V>());
                vectorToMte3Events_[slot] = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::V_MTE3>());
            } else {
                mte2ToScalarEvents_[slot] = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::MTE2_S>());
                scalarToMte3Events_[slot] = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::S_MTE3>());
            }
            if (UseScalarBufferEvents()) {
                mte3ToScalarEvents_[slot] = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::MTE3_S>());
            } else {
                mte3ToMte2Events_[slot] = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            }
        }
        if constexpr (HasSourceOperand || HasDestinationOperand || HasEdgeMap) {
            mte2ToScalarMetadataEvent_ = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::MTE2_S>());
        }
        if constexpr (UseBroadcast) {
            mte2ToScalarOffsetsEvent_ = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::MTE2_S>());
            if constexpr (IsFp32) {
                scalarToVectorOffsetsEvent_ = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::S_V>());
                scalarToVectorInputsEvent_ = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::S_V>());
                mte2ToVectorBroadcastEvent_ = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::MTE2_V>());
                vectorToMte2BroadcastEvent_ = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::V_MTE2>());
            } else {
                mte2ToScalarBroadcastEvent_ = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::MTE2_S>());
                scalarToMte2BroadcastEvent_ = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::S_MTE2>());
            }
        }
        if constexpr (IsFp32 && !UseBroadcast) {
            mte2ToScalarFeatureDrainEvent_ = static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::MTE2_S>());
        }
    }
    __aicore__ inline void ReleaseEvents(AscendC::TPipe &pipe) {
        if (IsDot()) {
            pipe.ReleaseEventID<AscendC::HardEvent::MTE3_S>(dotMte3ToScalarEvent_);
            pipe.ReleaseEventID<AscendC::HardEvent::MTE2_S>(dotMte2ToScalarEvent_);
            if constexpr (IsFp32) {
                pipe.ReleaseEventID<AscendC::HardEvent::V_S>(dotVectorToScalarEvent_);
                pipe.ReleaseEventID<AscendC::HardEvent::S_MTE3>(dotScalarToMte3Event_);
            }
        }
        for (uint32_t slot = 0; slot < BufferCount; ++slot) {
            if constexpr (IsFp32) {
                pipe.ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToVectorEvents_[slot]);
                pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(vectorToMte3Events_[slot]);
            } else {
                pipe.ReleaseEventID<AscendC::HardEvent::MTE2_S>(mte2ToScalarEvents_[slot]);
                pipe.ReleaseEventID<AscendC::HardEvent::S_MTE3>(scalarToMte3Events_[slot]);
            }
            if (UseScalarBufferEvents()) {
                pipe.ReleaseEventID<AscendC::HardEvent::MTE3_S>(mte3ToScalarEvents_[slot]);
            } else {
                pipe.ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Events_[slot]);
            }
        }
        if constexpr (HasSourceOperand || HasDestinationOperand || HasEdgeMap) {
            pipe.ReleaseEventID<AscendC::HardEvent::MTE2_S>(mte2ToScalarMetadataEvent_);
        }
        if constexpr (UseBroadcast) {
            pipe.ReleaseEventID<AscendC::HardEvent::MTE2_S>(mte2ToScalarOffsetsEvent_);
            if constexpr (IsFp32) {
                pipe.ReleaseEventID<AscendC::HardEvent::S_V>(scalarToVectorOffsetsEvent_);
                pipe.ReleaseEventID<AscendC::HardEvent::S_V>(scalarToVectorInputsEvent_);
                pipe.ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToVectorBroadcastEvent_);
                pipe.ReleaseEventID<AscendC::HardEvent::V_MTE2>(vectorToMte2BroadcastEvent_);
            } else {
                pipe.ReleaseEventID<AscendC::HardEvent::MTE2_S>(mte2ToScalarBroadcastEvent_);
                pipe.ReleaseEventID<AscendC::HardEvent::S_MTE2>(scalarToMte2BroadcastEvent_);
            }
        }
        if constexpr (IsFp32 && !UseBroadcast) {
            pipe.ReleaseEventID<AscendC::HardEvent::MTE2_S>(mte2ToScalarFeatureDrainEvent_);
        }
    }
    __aicore__ inline bool UseScalarBufferEvents() const {
        if constexpr (!IsFp32 || UseBroadcast) return true;
        if constexpr (AddOnly) return false;
        return !dotVectorOnly_;
    }
    __aicore__ inline void WaitForBuffer(uint32_t slot) {
        if (UseScalarBufferEvents()) AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalarEvents_[slot]);
        else AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Events_[slot]);
    }
    // MTE3 写回完成后允许该槽被后续输入搬运覆盖，与 WaitForBuffer 配对。
    __aicore__ inline void SetForBuffer(uint32_t slot) {
        if (UseScalarBufferEvents()) AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalarEvents_[slot]);
        else AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Events_[slot]);
    }
    __aicore__ inline void CopyIndices(uint64_t csrEdge) {
        if constexpr (HasDestinationOperand || HasEdgeMap) {
            if (csrEdge < indicesCacheBegin_ || csrEdge >= edgeCacheEnd_) {
                indicesCacheBegin_ = csrEdge;
                edgeCacheEnd_ = Min(edgeEnd_, csrEdge + indexCacheEdges_);
                const AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>((edgeCacheEnd_ - csrEdge) * sizeof(IdType)), 0, 0, 0};
                const AscendC::DataCopyPadExtParams<IdType> padParams = {false, 0, 0, 0};
                if constexpr (HasDestinationOperand) AscendC::DataCopyPad(indicesLocal_, indicesGm_[csrEdge], copyParams, padParams);
                if constexpr (HasEdgeMap) AscendC::DataCopyPad(edgeMapLocal_, edgeMapGm_[csrEdge], copyParams, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarMetadataEvent_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarMetadataEvent_);
            }
        }
    }
    // 对于源节点行 u 的边范围，返回 indptr[u+1]
    __aicore__ inline uint64_t GetSourceRowEdgeEnd(uint64_t sourceRow) {
        const uint64_t indptrIndex = sourceRow + 1;
        if (indptrIndex < indptrCacheBegin_ || indptrIndex >= indptrCacheEnd_) {
            indptrCacheBegin_ = indptrIndex / IndptrCacheEntryCount * IndptrCacheEntryCount;
            indptrCacheEnd_ = Min(numRows_ + 1, indptrCacheBegin_ + IndptrCacheEntryCount);
            const AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>((indptrCacheEnd_ - indptrCacheBegin_) * sizeof(IdType)), 0, 0, 0};
            AscendC::DataCopyPad(indptrLocal_, indptrGm_[indptrCacheBegin_], copyParams, AscendC::DataCopyPadExtParams<IdType>{false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarMetadataEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarMetadataEvent_);
        }
        return static_cast<uint64_t>(indptrLocal_.GetValue(indptrIndex - indptrCacheBegin_));
    }
    template <uint32_t Target>
    __aicore__ inline uint64_t GetFeatureRow(uint64_t csrEdge, uint32_t rowIndex, uint64_t tileSourceRowBegin) {
        if constexpr (Target == 0) return tileSourceRowBegin + rowIndex;
        else if constexpr (Target == 1) {
            if constexpr (HasEdgeMap) return static_cast<uint64_t>(edgeMapLocal_.GetValue(csrEdge - indicesCacheBegin_ + rowIndex));
            else return csrEdge + rowIndex;
        } else return static_cast<uint64_t>(indicesLocal_.GetValue(csrEdge - indicesCacheBegin_ + rowIndex));
    }
    __aicore__ inline void CopyFeatureRow(const AscendC::LocalTensor<T> &localData,
                                 const AscendC::GlobalTensor<T> &globalData, uint64_t globalElementOffset, uint32_t featureCount, uint32_t localRowStride) {
        const uint32_t paddingElements = localRowStride - featureCount;
        AscendC::DataCopyPad(localData, globalData[globalElementOffset], AscendC::DataCopyExtParams{1, featureCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0},
            AscendC::DataCopyPadExtParams<T>{paddingElements != 0, 0, static_cast<uint8_t>(paddingElements), 0});
    }
    // Copy 广播时所需要的特征值
    template <uint32_t Target>
    __aicore__ inline void CopyGatherNeedFeatureRows(const AscendC::LocalTensor<T> &localData, const AscendC::GlobalTensor<T> &globalData, uint32_t inputFeatureSize, uint64_t start, uint32_t featureCount, uint32_t localRowStride, uint64_t csrEdge, uint32_t rowCount, uint64_t tileSourceRowBegin) {
        if constexpr (Target == 0 || (Target == 1 && !HasEdgeMap)) {
            const uint64_t row = Target == 0 ? tileSourceRowBegin : csrEdge;
            if (start == 0 && featureCount == inputFeatureSize && localRowStride == featureCount) {
                AscendC::DataCopy(localData, globalData[row * inputFeatureSize], rowCount * featureCount);
                return;
            }
            const uint64_t rowGapBytes = (static_cast<uint64_t>(inputFeatureSize) - featureCount) * sizeof(T);
            AscendC::DataCopyPad(localData, globalData[row * inputFeatureSize + start], AscendC::DataCopyExtParams{static_cast<uint16_t>(rowCount), featureCount * static_cast<uint32_t>(sizeof(T)), static_cast<uint32_t>(rowGapBytes), 0, 0}, AscendC::DataCopyPadExtParams<T>{localRowStride != featureCount, 0, static_cast<uint8_t>(localRowStride - featureCount), 0});
            return;
        }
        if constexpr (Target != 0 && (Target == 2 || HasEdgeMap)) {
            if (featureCount == localRowStride && (inputFeatureSize * sizeof(T)) % 32 == 0 && (start * sizeof(T)) % 32 == 0) {
                uint32_t rowIndex = 0;
                for (; rowIndex + 4 <= rowCount; rowIndex += 4) {
                    const uint64_t row0 = GetFeatureRow<Target>(csrEdge, rowIndex, tileSourceRowBegin);
                    const uint64_t row1 = GetFeatureRow<Target>(csrEdge, rowIndex + 1, tileSourceRowBegin);
                    const uint64_t row2 = GetFeatureRow<Target>(csrEdge, rowIndex + 2, tileSourceRowBegin);
                    const uint64_t row3 = GetFeatureRow<Target>(csrEdge, rowIndex + 3, tileSourceRowBegin);
                    AscendC::DataCopy(localData[rowIndex * localRowStride], globalData[row0 * inputFeatureSize + start], featureCount);
                    AscendC::DataCopy(localData[(rowIndex + 1) * localRowStride], globalData[row1 * inputFeatureSize + start], featureCount);
                    AscendC::DataCopy(localData[(rowIndex + 2) * localRowStride], globalData[row2 * inputFeatureSize + start], featureCount);
                    AscendC::DataCopy(localData[(rowIndex + 3) * localRowStride], globalData[row3 * inputFeatureSize + start], featureCount);
                }
                for (; rowIndex < rowCount; ++rowIndex) {
                    const uint64_t row = GetFeatureRow<Target>(csrEdge, rowIndex, tileSourceRowBegin);
                    AscendC::DataCopy(localData[rowIndex * localRowStride], globalData[row * inputFeatureSize + start], featureCount);
                }
                return;
            }
        }
        for (uint32_t i = 0; i < rowCount; ++i) {
            CopyFeatureRow(localData[i * localRowStride], globalData, GetFeatureRow<Target>(csrEdge, i, tileSourceRowBegin) * inputFeatureSize + start, featureCount, localRowStride);
        }
    }

    __aicore__ inline BroadcastPlan PrepareBroadcastPlan(const AscendC::GlobalTensor<int64_t> &offsets, AscendC::TBuf<AscendC::TPosition::VECCALC> &offsetBuf, bool hasOffsetTensor, uint32_t inputFeatureSize) {
        BroadcastPlan plan;
        if (!hasOffsetTensor && inputFeatureSize == outFeatureSize_) return plan;
        auto elementOffsets = offsetBuf.Get<int64_t>();
        if (hasOffsetTensor) {
            AscendC::DataCopyPad(elementOffsets, offsets[featureBegin_], AscendC::DataCopyExtParams{1, featureCount_ * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0}, AscendC::DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarOffsetsEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarOffsetsEvent_);
        } else {
            for (uint32_t featureIndex = 0; featureIndex < featureCount_; ++featureIndex) elementOffsets.SetValue(featureIndex, 0);
        }
        uint64_t minInputOffset = static_cast<uint64_t>(elementOffsets.GetValue(0));
        uint64_t maxInputOffset = minInputOffset;
        bool isIdentity = inputFeatureSize == outFeatureSize_;
        for (uint32_t featureIndex = 0; featureIndex < featureCount_; ++featureIndex) {
            const uint64_t inputOffset = static_cast<uint64_t>(elementOffsets.GetValue(featureIndex));
            if (inputOffset < minInputOffset) minInputOffset = inputOffset;
            if (inputOffset > maxInputOffset) maxInputOffset = inputOffset;
            if (inputOffset != static_cast<uint64_t>(featureBegin_) + featureIndex) isIdentity = false;
        }
        if (isIdentity) return plan;
        plan.isIdentity = false;
        plan.minInputOffset = minInputOffset;
        const uint64_t inputSpan = maxInputOffset - minInputOffset + 1;
        plan.fitsTemp = inputSpan <= tempFeatureCapacity_;
        if (plan.fitsTemp) {
            plan.inputSpan = static_cast<uint32_t>(inputSpan);
            plan.tempRowStride = AlignedFeature(plan.inputSpan);
            // Gather API 可以操作 Float32 数据类型，准备 Gather 所需的临时缓冲区字节偏移
            if constexpr (IsFp32) {
                auto byteOffsets = offsetBuf.Get<uint32_t>();
                for (uint32_t featureIndex = 0; featureIndex < featureCount_; ++featureIndex) {
                    const uint64_t inputOffset = static_cast<uint64_t>(elementOffsets.GetValue(featureIndex));
                    byteOffsets.SetValue(featureIndex, static_cast<uint32_t>((inputOffset - minInputOffset) * sizeof(T)));
                }
                AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarToVectorOffsetsEvent_);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarToVectorOffsetsEvent_);
            }
        }
        return plan;
    }

    template <uint32_t Target>
    __aicore__ inline void CopyOperand(const AscendC::LocalTensor<T> &localData, const AscendC::GlobalTensor<T> &globalData, uint32_t inputFeatureSize, AscendC::TBuf<AscendC::TPosition::VECCALC> &offsetBuf, const BroadcastPlan &plan, uint64_t csrEdge, uint32_t edgeCount, uint64_t tileSourceRowBegin, uint32_t sourceRowCount) {
        const uint32_t rowCount = Target == 0 ? sourceRowCount : edgeCount;
        if constexpr (UseBroadcast) {
            // 广播时搬运操作数
            if (!plan.isIdentity) {
                auto offsets = offsetBuf.Get<int64_t>();
                if (!plan.fitsTemp) {
                    // 该 Tile 所涉及的源特征值跨度比较大，超出临时广播可以容纳的元素数量，需要结合 GetValue 和 SetValue 来赋值
                    for (uint32_t i = 0; i < rowCount; ++i) {
                        const uint64_t row = GetFeatureRow<Target>(csrEdge, i, tileSourceRowBegin);
                        for (uint32_t featureIndex = 0; featureIndex < featureCount_; ++featureIndex) 
                            localData.SetValue(i * localFeatureStride_ + featureIndex, globalData.GetValue(row * inputFeatureSize + static_cast<uint64_t>(offsets.GetValue(featureIndex))));
                    }
                    if constexpr (IsFp32) {
                        for (uint32_t i = 0; i < rowCount; ++i)
                            for (uint32_t featureIndex = featureCount_; featureIndex < localFeatureStride_; ++featureIndex) localData.SetValue(i * localFeatureStride_ + featureIndex, 0);
                    }
                    return;
                }
                auto temp = broadcastTempBuf_.Get<T>();
                CopyGatherNeedFeatureRows<Target>(temp, globalData, inputFeatureSize, plan.minInputOffset, plan.inputSpan, plan.tempRowStride, csrEdge, rowCount, tileSourceRowBegin);
                if constexpr (IsFp32) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVectorBroadcastEvent_);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVectorBroadcastEvent_);
                    if (featureCount_ != localFeatureStride_) {
                        AscendC::Duplicate(localData, 0.0F, rowCount * localFeatureStride_);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    auto byteOffsets = offsetBuf.Get<uint32_t>();
                    for (uint32_t i = 0; i < rowCount; ++i) AscendC::Gather(localData[i * localFeatureStride_], temp[i * plan.tempRowStride], byteOffsets, 0, featureCount_);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vectorToMte2BroadcastEvent_);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vectorToMte2BroadcastEvent_);
                } else {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarBroadcastEvent_);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarBroadcastEvent_);
                    for (uint32_t i = 0; i < rowCount; ++i)
                        for (uint32_t featureIndex = 0; featureIndex < featureCount_; ++featureIndex)
                            localData.SetValue(i * localFeatureStride_ + featureIndex, temp.GetValue(i * plan.tempRowStride + static_cast<uint32_t>(static_cast<uint64_t>(offsets.GetValue(featureIndex)) - plan.minInputOffset)));
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2BroadcastEvent_);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2BroadcastEvent_);
                }
                return;
            }
        }
        // 非广播时搬运操作数
        if constexpr (Target == 0 || (Target == 1 && !HasEdgeMap)) {
            const uint64_t row = Target == 0 ? tileSourceRowBegin : csrEdge;
            if (featureBegin_ == 0 && featureCount_ == inputFeatureSize && localFeatureStride_ == featureCount_) {
                AscendC::DataCopy(localData, globalData[row * inputFeatureSize], rowCount * featureCount_);
                return;
            }
            const uint64_t rowGapBytes = (static_cast<uint64_t>(inputFeatureSize) - featureCount_) * sizeof(T);
            AscendC::DataCopyPad(localData, globalData[row * inputFeatureSize + featureBegin_],
                AscendC::DataCopyExtParams{static_cast<uint16_t>(rowCount), featureCount_ * static_cast<uint32_t>(sizeof(T)), static_cast<uint32_t>(rowGapBytes), 0, 0},
                AscendC::DataCopyPadExtParams<T>{localFeatureStride_ != featureCount_, 0, static_cast<uint8_t>(localFeatureStride_ - featureCount_), 0});
            return;
        }
        if constexpr (Target != 0 && (Target == 2 || HasEdgeMap)) {
            if (featureCount_ == localFeatureStride_ && (inputFeatureSize * sizeof(T)) % 32 == 0 && (featureBegin_ * sizeof(T)) % 32 == 0) {
                uint32_t rowIndex = 0;
                for (; rowIndex + 4 <= rowCount; rowIndex += 4) {
                    const uint64_t row0 = GetFeatureRow<Target>(csrEdge, rowIndex, tileSourceRowBegin);
                    const uint64_t row1 = GetFeatureRow<Target>(csrEdge, rowIndex + 1, tileSourceRowBegin);
                    const uint64_t row2 = GetFeatureRow<Target>(csrEdge, rowIndex + 2, tileSourceRowBegin);
                    const uint64_t row3 = GetFeatureRow<Target>(csrEdge, rowIndex + 3, tileSourceRowBegin);
                    AscendC::DataCopy(localData[rowIndex * localFeatureStride_], globalData[row0 * inputFeatureSize + featureBegin_], featureCount_);
                    AscendC::DataCopy(localData[(rowIndex + 1) * localFeatureStride_], globalData[row1 * inputFeatureSize + featureBegin_], featureCount_);
                    AscendC::DataCopy(localData[(rowIndex + 2) * localFeatureStride_], globalData[row2 * inputFeatureSize + featureBegin_], featureCount_);
                    AscendC::DataCopy(localData[(rowIndex + 3) * localFeatureStride_], globalData[row3 * inputFeatureSize + featureBegin_], featureCount_);
                }
                for (; rowIndex < rowCount; ++rowIndex) {
                    const uint64_t row = GetFeatureRow<Target>(csrEdge, rowIndex, tileSourceRowBegin);
                    AscendC::DataCopy(localData[rowIndex * localFeatureStride_], globalData[row * inputFeatureSize + featureBegin_], featureCount_);
                }
                return;
            }
        }
        const uint32_t paddingElements = localFeatureStride_ - featureCount_;
        for (uint32_t i = 0; i < rowCount; ++i) {
            const uint64_t row = GetFeatureRow<Target>(csrEdge, i, tileSourceRowBegin);
            AscendC::DataCopyPad(localData[i * localFeatureStride_], globalData[row * inputFeatureSize + featureBegin_],
                AscendC::DataCopyExtParams{1, featureCount_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<T>{paddingElements != 0, 0, static_cast<uint8_t>(paddingElements), 0});
        }
    }

    __aicore__ inline bool IsCopy() const {
        if constexpr (AddOnly) return false;
        return operation_ == SDDMMCsrOp::CopyLhs || operation_ == SDDMMCsrOp::CopyRhs;
    }
    __aicore__ inline bool IsDot() const {
        if constexpr (AddOnly) return false;
        return operation_ == SDDMMCsrOp::Dot;
    }
    __aicore__ inline uint64_t ComputeScalar(uint64_t lhs, uint64_t rhs) const {
        if constexpr (SwapInputsHs) {
            const uint64_t saved = lhs;
            lhs = rhs;
            rhs = saved;
        }
        if constexpr (AddOnly) return AddFp64Bits(lhs, rhs);
        switch (operation_) {
            case SDDMMCsrOp::Sub: return SubFp64Bits(lhs, rhs);
            case SDDMMCsrOp::Mul:
            case SDDMMCsrOp::Dot: return MulFp64Bits(lhs, rhs);
            case SDDMMCsrOp::Div: return DivFp64Bits(lhs, rhs);
            default: return AddFp64Bits(lhs, rhs);
        }
    }
    __aicore__ inline void ComputeVector(const AscendC::LocalTensor<T> &out,
        const AscendC::LocalTensor<T> &lhs, const AscendC::LocalTensor<T> &rhs, uint32_t count) {
        if constexpr (AddOnly) {
            AscendC::Add(out, lhs, rhs, count);
        } else {
            const auto left = SwapInputsHs ? rhs : lhs;
            const auto right = SwapInputsHs ? lhs : rhs;
            switch (operation_) {
                case SDDMMCsrOp::Sub: AscendC::Sub(out, left, right, count); break;
                case SDDMMCsrOp::Mul:
                case SDDMMCsrOp::Dot: AscendC::Mul(out, left, right, count); break;
                case SDDMMCsrOp::Div: AscendC::Div(out, left, right, count); break;
                default: AscendC::Add(out, left, right, count); break;
            }
        }
    }
    __aicore__ inline void ComputeVector(const AscendC::LocalTensor<T> &out,
        const AscendC::LocalTensor<T> &lhs, const AscendC::LocalTensor<T> &rhs,
        uint32_t mask, uint8_t repeats, AscendC::BinaryRepeatParams params) {
        if constexpr (AddOnly) {
            AscendC::Add<T, false>(out, lhs, rhs, mask, repeats, params);
        } else {
            const auto left = SwapInputsHs ? rhs : lhs;
            const auto right = SwapInputsHs ? lhs : rhs;
            if constexpr (SwapInputsHs) {
                const uint8_t saved = params.src0RepStride;
                params.src0RepStride = params.src1RepStride;
                params.src1RepStride = saved;
            }
            switch (operation_) {
                case SDDMMCsrOp::Sub: AscendC::Sub<T, false>(out, left, right, mask, repeats, params); break;
                case SDDMMCsrOp::Mul:
                case SDDMMCsrOp::Dot: AscendC::Mul<T, false>(out, left, right, mask, repeats, params); break;
                case SDDMMCsrOp::Div: AscendC::Div<T, false>(out, left, right, mask, repeats, params); break;
                default: AscendC::Add<T, false>(out, left, right, mask, repeats, params); break;
            }
        }
    }
    __aicore__ inline void ComputeTile(const AscendC::LocalTensor<T> &out, const AscendC::LocalTensor<T> &lhs, const AscendC::LocalTensor<T> &rhs, uint32_t edgeCount) {
        // Host normalizes both targets to the selected operand for copy.
        // The edge/destination path already owns its output in rhsTile.
        if (IsCopy()) {
            if constexpr (BothOperandsUseSource) {
                for (uint32_t i = 0; i < edgeCount; ++i) {
                    if constexpr (IsFp32) AscendC::DataCopy(out[i * localFeatureStride_], rhs, localFeatureStride_);
                    else for (uint32_t j = 0; j < featureCount_; ++j)
                        out.SetValue(i * localFeatureStride_ + j, rhs.GetValue(j));
                }
            }
            return;
        }
        if constexpr (IsFp32) {
            if constexpr (!HasSourceOperand) {
                ComputeVector(out, lhs, rhs, edgeCount * localFeatureStride_);
                return;
            }
            if (localFeatureStride_ / 8 < 256) {
                const uint8_t strideBlocks = static_cast<uint8_t>(localFeatureStride_ / 8);
                // Source rows are reused with a zero repeat stride.
                const AscendC::BinaryRepeatParams repeatParams = {1, 1, 1, strideBlocks, 0, static_cast<uint8_t>(BothOperandsUseSource ? 0 : strideBlocks)};
                // Generic dot reduction changes the vector mask between tiles.
                if constexpr (UseBroadcast || !AddOnly) {
                    AscendC::SetMaskNorm();
                    AscendC::SetVectorMask<T>(static_cast<int32_t>(featureCount_ < 64 ? featureCount_ : 64));
                }
                const uint32_t fullVectorFeatures = featureCount_ / 64 * 64;
                for (uint32_t featureIndex = 0; featureIndex < fullVectorFeatures; featureIndex += 64) {
                    ComputeVector(out[featureIndex], lhs[featureIndex], rhs[featureIndex], 64, static_cast<uint8_t>(edgeCount), repeatParams);
                }
                const uint32_t tailFeatures = featureCount_ - fullVectorFeatures;
                if (tailFeatures != 0) {
                    if (fullVectorFeatures != 0) AscendC::SetVectorMask<T>(static_cast<int32_t>(tailFeatures));
                    ComputeVector(out[fullVectorFeatures], lhs[fullVectorFeatures], rhs[fullVectorFeatures], tailFeatures, static_cast<uint8_t>(edgeCount), repeatParams);
                    if (fullVectorFeatures != 0) AscendC::SetVectorMask<T>(64);
                }
                return;
            }
            for (uint32_t i = 0; i < edgeCount; ++i) {
                ComputeVector(out[i * localFeatureStride_], lhs, rhs[BothOperandsUseSource ? 0 : i * localFeatureStride_], featureCount_);
            }
        } else {
            for (uint32_t featureIndex = 0; featureIndex < featureCount_; ++featureIndex) {
                if constexpr (BothOperandsUseSource) {
                    const uint64_t result = ComputeScalar(lhs.GetValue(featureIndex), rhs.GetValue(featureIndex));
                    for (uint32_t i = 0; i < edgeCount; ++i) out.SetValue(i * localFeatureStride_ + featureIndex, result);
                } else {
                    for (uint32_t i = 0; i < edgeCount; ++i) {
                        const uint64_t lhsBits = lhs.GetValue((HasSourceOperand ? 0 : i * localFeatureStride_) + featureIndex);
                        const uint64_t rhsBits = rhs.GetValue(i * localFeatureStride_ + featureIndex);
                        out.SetValue(i * localFeatureStride_ + featureIndex, ComputeScalar(lhsBits, rhsBits));
                    }
                }
            }
        }
    }
    __aicore__ inline uint64_t GetOutputEdge(uint64_t csrEdge, uint32_t row) const {
        if constexpr (HasEdgeMap) return static_cast<uint64_t>(edgeMapLocal_.GetValue(csrEdge - indicesCacheBegin_ + row));
        return csrEdge + row;
    }
    // Reduce a complete FP32 feature row into consecutive output elements.
    // Padded rows are still required for arbitrary edge-map scatter addresses.
    __aicore__ inline bool ReduceDotPacked(const AscendC::LocalTensor<T> &out,
        const AscendC::LocalTensor<T> &products, uint32_t edgeCount) {
        if constexpr (!IsFp32 || AddOnly) return false;
        if (!dotVectorOnly_ || localFeatureStride_ != featureCount_) return false;
        const uint32_t heads = outFeatureSize_ / reduceSize_;
        if constexpr (HasEdgeMap) {
            if (heads % 8 != 0) return false;
        }
        auto temp = dotTempBuf_.Get<T>();
        if (reduceSize_ == 16) {
            // Each block sum covers eight input elements. Adjacent pairs of
            // block sums produce all 16-wide heads in one contiguous layout.
            const uint32_t elements = edgeCount * featureCount_;
            uint32_t offset = 0;
            AscendC::PipeBarrier<PIPE_V>();
            while (elements - offset >= 64) {
                const uint32_t repeats = static_cast<uint32_t>(Min((elements - offset) / 64, 255));
                AscendC::BlockReduceSum<T, true>(temp[offset / 8], products[offset], repeats, 64, 1, 1, 8);
                offset += repeats * 64;
            }
            if (offset < elements)
                AscendC::BlockReduceSum<T, true>(temp[offset / 8], products[offset], 1, elements - offset, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            const uint32_t blockSums = elements / 8;
            offset = 0;
            // PairReduceSum advances its destination in 128-byte units;
            // a full float repeat produces exactly 32 consecutive values.
            while (blockSums - offset >= 64) {
                const uint32_t repeats = static_cast<uint32_t>(Min((blockSums - offset) / 64, 255));
                AscendC::PairReduceSum<T, true>(out[offset / 2], temp[offset], repeats, 64, 1, 1, 8);
                offset += repeats * 64;
            }
            if (offset < blockSums)
                AscendC::PairReduceSum<T, true>(out[offset / 2], temp[offset], 1, blockSums - offset, 1, 1, 8);
        } else if (heads == 1) {
            // WholeReduceSum's destination repeat stride is in elements, so
            // single-head results need no per-edge 32-byte padding in UB.
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t offset = 0; offset < reduceSize_; offset += 64) {
                const uint32_t count = static_cast<uint32_t>(Min(reduceSize_ - offset, 64));
                if (offset == 0) {
                    AscendC::WholeReduceSum<T, true>(out, products, count, edgeCount, 1, 1, localFeatureStride_ / 8);
                } else {
                    AscendC::WholeReduceSum<T, true>(temp, products[offset], count, edgeCount, 1, 1, localFeatureStride_ / 8);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(out, out, temp, edgeCount);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        } else {
            return false;
        }
        dotFeatureStride_ = heads;
        dotScalarOutput_ = false;
        return true;
    }
    __aicore__ inline void ReduceDot(const AscendC::LocalTensor<T> &out,
        const AscendC::LocalTensor<T> &products, uint64_t csrEdge, uint32_t edgeCount) {
        const uint32_t firstOffset = featureBegin_ % reduceSize_;
        dotFeatureCount_ = (firstOffset + featureCount_ - 1) / reduceSize_ + 1;
        dotFeatureStride_ = AlignedFeature(dotFeatureCount_);
        dotScalarOutput_ = true;
        if constexpr (IsFp32) {
            if (ReduceDotPacked(out, products, edgeCount)) return;
            // The common aligned path reduces each head across all tile edges.
            // Large or unaligned reductions use the scalar path below, including
            // partial heads crossing feature tiles, without reading GM padding.
            if (reduceSize_ % 8 == 0 && localFeatureStride_ / 8 < 256 && dotFeatureStride_ / 8 < 256) {
                AscendC::PipeBarrier<PIPE_V>();
                auto temp = dotTempBuf_.Get<T>();
                uint32_t inputOffset = 0;
                for (uint32_t group = 0; group < dotFeatureCount_; ++group) {
                    const uint32_t count = static_cast<uint32_t>(Min(featureCount_ - inputOffset, reduceSize_ - (featureBegin_ + inputOffset) % reduceSize_));
                    for (uint32_t offset = 0; offset < count; offset += 64) {
                        const uint32_t chunk = static_cast<uint32_t>(Min(count - offset, 64));
                        if (offset == 0) {
                            AscendC::WholeReduceSum<T, true>(out[group], products[inputOffset], chunk, edgeCount, dotFeatureStride_, 1, localFeatureStride_ / 8);
                        } else {
                            AscendC::WholeReduceSum<T, true>(temp[group % 8], products[inputOffset + offset], chunk, edgeCount, 8, 1, localFeatureStride_ / 8);
                            AscendC::PipeBarrier<PIPE_V>();
                            const uint8_t stride = static_cast<uint8_t>(dotFeatureStride_ / 8);
                            uint64_t mask[2] = {1ULL << (group % 8), 0};
                            const uint32_t blockBegin = group / 8 * 8;
                            AscendC::Add<T, true>(out[blockBegin], out[blockBegin], temp, mask, static_cast<uint8_t>(edgeCount), AscendC::BinaryRepeatParams{1, 1, 1, stride, stride, 1});
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                    }
                    inputOffset += count;
                }
                dotScalarOutput_ = false;
            }
            if (dotScalarOutput_ || firstOffset != 0) {
                AscendC::SetFlag<AscendC::HardEvent::V_S>(dotVectorToScalarEvent_);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(dotVectorToScalarEvent_);
            }
        }
        if (dotScalarOutput_) {
            for (uint32_t row = 0; row < edgeCount; ++row) {
                uint32_t inputOffset = 0;
                for (uint32_t group = 0; group < dotFeatureCount_; ++group) {
                    const uint32_t count = static_cast<uint32_t>(Min(featureCount_ - inputOffset, reduceSize_ - (featureBegin_ + inputOffset) % reduceSize_));
                    T sum = 0;
                    for (uint32_t k = 0; k < count; ++k) {
                        const T value = products.GetValue(row * localFeatureStride_ + inputOffset + k);
                        if constexpr (IsFp32) sum += value;
                        else sum = AddFp64Bits(sum, value);
                    }
                    out.SetValue(row * dotFeatureStride_ + group, sum);
                    inputOffset += count;
                }
            }
        }
        if (firstOffset != 0) {
            const uint32_t outputWidth = outFeatureSize_ / reduceSize_;
            // Use MTE2 rather than scalar GM reads: MTE3 writes do not invalidate
            // the scalar data cache when a head spans three or more tiles.
            auto previousTile = dotTempBuf_.Get<T>();
            constexpr uint32_t previousStride = 32 / sizeof(T);
            for (uint32_t row = 0; row < edgeCount; ++row)
                CopyFeatureRow(previousTile[row * previousStride], outGm_,
                    GetOutputEdge(csrEdge, row) * outputWidth + featureBegin_ / reduceSize_, 1, previousStride);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(dotMte2ToScalarEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(dotMte2ToScalarEvent_);
            for (uint32_t row = 0; row < edgeCount; ++row) {
                const T previous = previousTile.GetValue(row * previousStride);
                const T partial = out.GetValue(row * dotFeatureStride_);
                if constexpr (IsFp32) out.SetValue(row * dotFeatureStride_, previous + partial);
                else out.SetValue(row * dotFeatureStride_, AddFp64Bits(previous, partial));
            }
            dotScalarOutput_ = true;
        }
    }
    __aicore__ inline void CopyOutDot(const AscendC::LocalTensor<T> &data, uint64_t csrEdge, uint32_t edgeCount) {
        const uint32_t outputWidth = outFeatureSize_ / reduceSize_;
        const uint32_t outputBegin = featureBegin_ / reduceSize_;
        if constexpr (!HasEdgeMap) {
            if (outputBegin == 0 && dotFeatureCount_ == outputWidth && dotFeatureStride_ == outputWidth) {
                AscendC::DataCopyPad(outGm_[csrEdge * outputWidth], data,
                    AscendC::DataCopyExtParams{1, edgeCount * outputWidth * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
                return;
            }
            const uint32_t rowGapBytes = (outputWidth - dotFeatureCount_) * sizeof(T);
            AscendC::DataCopyPad(outGm_[csrEdge * outputWidth + outputBegin], data,
                AscendC::DataCopyExtParams{static_cast<uint16_t>(edgeCount), dotFeatureCount_ * static_cast<uint32_t>(sizeof(T)), 0, rowGapBytes, 0});
        } else {
            for (uint32_t row = 0; row < edgeCount; ++row)
                AscendC::DataCopyPad(outGm_[GetOutputEdge(csrEdge, row) * outputWidth + outputBegin], data[row * dotFeatureStride_],
                    AscendC::DataCopyExtParams{1, dotFeatureCount_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
        }
    }
    __aicore__ inline void CopyOut(const AscendC::LocalTensor<T> &data, uint64_t csrEdge, uint32_t edgeCount) {
        if constexpr (!HasEdgeMap) {
            if (featureBegin_ == 0 && featureCount_ == outFeatureSize_ && featureCount_ == localFeatureStride_) {
                AscendC::DataCopyPad(outGm_[csrEdge * outFeatureSize_], data, AscendC::DataCopyExtParams{1, edgeCount * featureCount_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
                return;
            }
            const uint64_t rowGapBytes = (static_cast<uint64_t>(outFeatureSize_) - featureCount_) * sizeof(T);
            AscendC::DataCopyPad(outGm_[csrEdge * outFeatureSize_ + featureBegin_], data, AscendC::DataCopyExtParams{static_cast<uint16_t>(edgeCount), featureCount_ * static_cast<uint32_t>(sizeof(T)), 0, static_cast<uint32_t>(rowGapBytes), 0});
            return;
        }
        for (uint32_t i = 0; i < edgeCount; ++i) {
            uint64_t outputEdge = csrEdge + i;
            if constexpr (HasEdgeMap) outputEdge = static_cast<uint64_t>(edgeMapLocal_.GetValue(csrEdge - indicesCacheBegin_ + i));
            AscendC::DataCopyPad(outGm_[outputEdge * outFeatureSize_ + featureBegin_], data[i * localFeatureStride_], AscendC::DataCopyExtParams{1, featureCount_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
        }
    }

    AscendC::GlobalTensor<T> lhsGm_, rhsGm_, outGm_;
    AscendC::GlobalTensor<IdType> indptrGm_, indicesGm_, edgeMapGm_;
    AscendC::GlobalTensor<int64_t> lhsOffsetsGm_, rhsOffsetsGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lhsBuf_, rhsBuf_, indptrBuf_, indicesBuf_, edgeMapBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> broadcastTempBuf_, lhsOffsetBuf_, rhsOffsetBuf_;
    AscendC::LocalTensor<T> lhsLocal_, rhsLocal_;
    AscendC::LocalTensor<IdType> indptrLocal_, indicesLocal_, edgeMapLocal_;
    BroadcastPlan lhsBroadcastPlan_, rhsBroadcastPlan_;
    uint64_t numRows_, edgeBegin_, edgeEnd_;
    uint64_t indicesCacheBegin_ = 0, edgeCacheEnd_ = 0;  // 一次性无法将边所有 indice Copy 进去，每次只能 Copy 一部分
    uint64_t indptrCacheBegin_ = 0, indptrCacheEnd_ = 0;  // 一次性无法将边所有 indptr Copy 进去，每次只能 Copy 一部分
    uint32_t tileEdges_;           // 单个计算 tile 的最大边数。
    uint32_t outFeatureSize_;      // 每条边的完整输出特征数。
    uint32_t tileFeatureSize_;     // CPU 按 32B 对齐的特征 tile 容量。
    uint32_t featureBegin_;        // 当前特征块在完整输出行中的起始下标。
    uint32_t featureCount_;        // 当前特征块的实际特征数。
    uint32_t localFeatureStride_;  // 当前特征块在 UB 中每行对齐后的元素数。
    uint32_t lhsFeatureSize_, rhsFeatureSize_;
    uint32_t lhsSlotElements_, rhsSlotElements_;  // 每个双缓冲槽占用的元素数。
    uint32_t indexCacheEdges_;         // 一次缓存 indices/edge_map 的最大边数。
    uint32_t tempFeatureCapacity_;   // 广播 temp 每行的最大输入特征数。
    // Each double-buffer slot owns its input, output, and reuse event IDs.
    int32_t mte2ToVectorEvents_[BufferCount];
    int32_t mte2ToScalarEvents_[BufferCount];
    int32_t vectorToMte3Events_[BufferCount];
    int32_t scalarToMte3Events_[BufferCount];
    int32_t mte3ToScalarEvents_[BufferCount];
    int32_t mte3ToMte2Events_[BufferCount];
    int32_t mte2ToScalarMetadataEvent_;
    int32_t mte2ToScalarOffsetsEvent_, scalarToVectorOffsetsEvent_;
    int32_t scalarToVectorInputsEvent_;
    int32_t mte2ToVectorBroadcastEvent_, vectorToMte2BroadcastEvent_;
    int32_t mte2ToScalarBroadcastEvent_, scalarToMte2BroadcastEvent_;
    int32_t mte2ToScalarFeatureDrainEvent_;
    bool hasLhsOffset_, hasRhsOffset_;
    SDDMMCsrOp operation_;
    uint32_t reduceSize_, dotSlotElements_, dotFeatureCount_, dotFeatureStride_;
    bool dotScalarOutput_ = false;
    bool dotVectorOnly_ = false;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dotOutBuf_, dotTempBuf_;
    int32_t dotVectorToScalarEvent_, dotScalarToMte3Event_, dotMte3ToScalarEvent_, dotMte2ToScalarEvent_;
};

// Preserve direct-launch symbols; add retains its compile-time specialization.
#define SDDMMCSR_ENTRY(Name, StorageType, IdType, LhsTarget, RhsTarget, UseBroadcast, HasEdgeMap, AddOnly) \
  extern "C" __global__ __aicore__ void Name( \
      GM_ADDR indptr, GM_ADDR indices, GM_ADDR lhs, GM_ADDR rhs, \
      GM_ADDR edgeMap, GM_ADDR lhsOffset, GM_ADDR rhsOffset, GM_ADDR out, \
      GM_ADDR tiling) { \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); \
    SDDMMCsrTilingData tilingData; \
    AscendC::GlobalTensor<uint32_t> tilingGm; \
    tilingGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tiling)); \
    auto* words = reinterpret_cast<uint32_t*>(&tilingData); \
    for (uint32_t i = 0; i < sizeof(tilingData) / sizeof(uint32_t); ++i) \
      words[i] = tilingGm.GetValue(i); \
    AscendC::TPipe pipe; \
    SDDMMCsrKernel<StorageType, IdType, LhsTarget, RhsTarget, UseBroadcast, HasEdgeMap, AddOnly> kernel; \
    kernel.Run(pipe, indptr, indices, lhs, rhs, edgeMap, lhsOffset, rhsOffset, out, tilingData); \
  }

#define SDDMMCSR_ENTRIES(Key, Lhs, Rhs, Bcast, Map) \
  SDDMMCSR_ENTRY(sddmm_csr_add_f32_i32_##Key, float, int32_t, Lhs, Rhs, Bcast, Map, true) \
  SDDMMCSR_ENTRY(sddmm_csr_add_f32_i64_##Key, float, int64_t, Lhs, Rhs, Bcast, Map, true) \
  SDDMMCSR_ENTRY(sddmm_csr_add_f64_i32_##Key, uint64_t, int32_t, Lhs, Rhs, Bcast, Map, true) \
  SDDMMCSR_ENTRY(sddmm_csr_add_f64_i64_##Key, uint64_t, int64_t, Lhs, Rhs, Bcast, Map, true) \
  SDDMMCSR_ENTRY(sddmm_csr_f32_i32_##Key, float, int32_t, Lhs, Rhs, Bcast, Map, false) \
  SDDMMCSR_ENTRY(sddmm_csr_f32_i64_##Key, float, int64_t, Lhs, Rhs, Bcast, Map, false) \
  SDDMMCSR_ENTRY(sddmm_csr_f64_i32_##Key, uint64_t, int32_t, Lhs, Rhs, Bcast, Map, false) \
  SDDMMCSR_ENTRY(sddmm_csr_f64_i64_##Key, uint64_t, int64_t, Lhs, Rhs, Bcast, Map, false)

SDDMMCSR_FOR_EACH_KEY(SDDMMCSR_ENTRIES)

#undef SDDMMCSR_ENTRIES
#undef SDDMMCSR_ENTRY
