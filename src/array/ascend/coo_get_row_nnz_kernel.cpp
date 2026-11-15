#include "kernel_operator.h"

#include <cstdint>

using namespace AscendC;

namespace {

constexpr uint32_t TileLength = 256;
constexpr uint32_t MaskBytes = TileLength * 2 / 8;
constexpr uint32_t FlushElements = 256;
constexpr uint32_t BufferNum = 2;

// Keep the field order identical to coo_get_row_nnz.cc.
struct COOGetRowNNZTiling {
  uint64_t nnz;
  uint64_t queryCount;
};

template <typename IdType>
class KernelCOOGetRowNNZ {
 public:
  __aicore__ inline void Init(GM_ADDR cooRows, GM_ADDR queryRows, GM_ADDR output, GM_ADDR tiling, TPipe* pipe) {
    pipe_ = pipe;
    const __gm__ COOGetRowNNZTiling* tilingData = reinterpret_cast<const __gm__ COOGetRowNNZTiling*>(tiling);
    nnz_ = tilingData->nnz;
    queryCount_ = tilingData->queryCount;

    cooRowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(cooRows), nnz_);
    outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(output), queryCount_);
    queryRows_ = reinterpret_cast<const __gm__ IdType*>(queryRows);

    pipe_->InitBuffer(inputBuffer_, BufferNum * TileLength * sizeof(IdType));
    pipe_->InitBuffer(maskLowBuffer_, MaskBytes);
    pipe_->InitBuffer(maskHighBuffer_, MaskBytes);
    pipe_->InitBuffer(outputBuffer_, FlushElements * sizeof(IdType));

    vectorToScalarEvent_ = static_cast<int32_t>(pipe_->AllocEventID<HardEvent::V_S>());
    scalarToVectorEvent_ = static_cast<int32_t>(pipe_->AllocEventID<HardEvent::S_V>());
    scalarToMte3Event_ = static_cast<int32_t>(pipe_->AllocEventID<HardEvent::S_MTE3>());
    mte3ToScalarEvent_ = static_cast<int32_t>(pipe_->AllocEventID<HardEvent::MTE3_S>());
    for (uint32_t slot = 0; slot < BufferNum; ++slot) {
      mte2ToVectorEvents_[slot] = static_cast<int32_t>(pipe_->AllocEventID<HardEvent::MTE2_V>());
    }
  }

  __aicore__ inline void Process() {
    if (queryCount_ == 0) {
      return;
    }

    const uint32_t blockIndex = GetBlockIdx();
    const uint32_t blockCount = GetBlockNum();
    const uint64_t queryBegin = static_cast<uint64_t>(queryCount_) * blockIndex / blockCount;
    const uint64_t queryEnd = static_cast<uint64_t>(queryCount_) * (blockIndex + 1) / blockCount;

    LocalTensor<IdType> outputLocal = outputBuffer_.Get<IdType>();

    uint32_t resultBlockIndex = 0;
    uint64_t outputBegin = queryBegin;

    for (uint64_t q = queryBegin; q < queryEnd; ++q) {
      const IdType target = queryRows_[q];
      const uint64_t count = Compute(target);
      outputLocal.SetValue(resultBlockIndex, static_cast<IdType>(count));
      ++resultBlockIndex;
      if (resultBlockIndex == FlushElements) {
        CopyOut(outputLocal, outputBegin, resultBlockIndex);
        outputBegin += resultBlockIndex;
        resultBlockIndex = 0;
      }
    }

    if (resultBlockIndex != 0) {
      CopyOut(outputLocal, outputBegin, resultBlockIndex);
    }

    PipeBarrier<PIPE_ALL>();
    pipe_->ReleaseEventID<HardEvent::V_S>(vectorToScalarEvent_);
    pipe_->ReleaseEventID<HardEvent::S_V>(scalarToVectorEvent_);
    pipe_->ReleaseEventID<HardEvent::S_MTE3>(scalarToMte3Event_);
    pipe_->ReleaseEventID<HardEvent::MTE3_S>(mte3ToScalarEvent_);
    for (uint32_t slot = 0; slot < BufferNum; ++slot) {
      pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVectorEvents_[slot]);
    }
  }

 private:
  __aicore__ inline uint64_t Compute(IdType target) {
    uint64_t total = 0;

    if (nnz_ == 0) {
      return total;
    }

    const uint64_t tileCount = 1 + (nnz_ - 1) / TileLength;
    CopyIn(0);

    for (uint64_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
      const uint64_t nextTile = tileIndex + 1;
      if (nextTile < tileCount) {
        CopyIn(nextTile);
      }

      total += ComputeTile(tileIndex, target);
    }

    return total;
  }

  __aicore__ inline void CopyIn(uint64_t tileIndex) {
    const uint64_t offset = tileIndex * TileLength;
    const uint64_t remaining = nnz_ - offset;
    const uint32_t validCount = static_cast<uint32_t>(remaining < TileLength ? remaining : TileLength);
    const uint32_t slot = static_cast<uint32_t>(tileIndex % BufferNum);
    const uint32_t inputOffset = slot * TileLength * sizeof(IdType);
    LocalTensor<IdType> input = inputBuffer_.GetWithOffset<IdType>(TileLength, inputOffset);

    const DataCopyExtParams copyParams{1, static_cast<uint32_t>(validCount * sizeof(IdType)), 0, 0, 0};
    const DataCopyPadExtParams<IdType> padParams{false, 0, 0, static_cast<IdType>(0)};
    DataCopyPad(input, cooRowsGm_[offset], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(mte2ToVectorEvents_[slot]);
  }

  __aicore__ inline uint32_t ComputeTile(uint64_t tileIndex, IdType target) {
    const uint64_t offset = tileIndex * TileLength;
    const uint64_t remaining = nnz_ - offset;
    const uint32_t validCount = static_cast<uint32_t>(remaining < TileLength ? remaining : TileLength);
    const uint32_t slot = static_cast<uint32_t>(tileIndex % BufferNum);
    const uint32_t inputOffset = slot * TileLength * sizeof(IdType);
    LocalTensor<IdType> input = inputBuffer_.GetWithOffset<IdType>(TileLength, inputOffset);

    WaitFlag<HardEvent::MTE2_V>(mte2ToVectorEvents_[slot]);
    return CountMatches(input, target, validCount);
  }

  __aicore__ inline uint32_t CountMatches(LocalTensor<IdType> input, IdType target, uint32_t validCount) {
    LocalTensor<uint8_t> maskLow = maskLowBuffer_.Get<uint8_t>();
    uint32_t count = 0;
    constexpr uint32_t bitsPerMaskWord = sizeof(uint64_t) * 8;
    if constexpr (sizeof(IdType) == 4) {
      CompareScalar(maskLow, input, target, CMPMODE::EQ, TileLength);

      SetFlag<HardEvent::V_S>(vectorToScalarEvent_);
      WaitFlag<HardEvent::V_S>(vectorToScalarEvent_);

      LocalTensor<uint64_t> maskWords = maskLow.ReinterpretCast<uint64_t>();
      const uint32_t wordCount = (validCount + bitsPerMaskWord - 1) / bitsPerMaskWord;
      for (uint32_t i = 0; i < wordCount; ++i) {
        uint64_t bits = maskWords.GetValue(i);
        const uint32_t first = i * bitsPerMaskWord;
        const uint32_t remaining = validCount - first;
        const uint32_t active = remaining < bitsPerMaskWord ? remaining : bitsPerMaskWord;
        if (active < bitsPerMaskWord) {
          bits &= (1ULL << active) - 1ULL;
        }
        count += static_cast<uint32_t>(GetBitCount<1>(bits));
      }
    } else {
      LocalTensor<int32_t> words = input.template ReinterpretCast<int32_t>();
      LocalTensor<uint8_t> maskHigh = maskHighBuffer_.Get<uint8_t>();

      const uint64_t targetBits = static_cast<uint64_t>(target);
      const int32_t targetLow = static_cast<int32_t>(static_cast<uint32_t>(targetBits));
      const int32_t targetHigh = static_cast<int32_t>(static_cast<uint32_t>(targetBits >> 32));

      CompareScalar(maskLow, words, targetLow, CMPMODE::EQ, TileLength * 2);
      CompareScalar(maskHigh, words, targetHigh, CMPMODE::EQ, TileLength * 2);

      SetFlag<HardEvent::V_S>(vectorToScalarEvent_);
      WaitFlag<HardEvent::V_S>(vectorToScalarEvent_);

      LocalTensor<uint64_t> maskLowWords = maskLow.ReinterpretCast<uint64_t>();
      LocalTensor<uint64_t> maskHighWords = maskHigh.ReinterpretCast<uint64_t>();
      constexpr uint32_t elementsPerMaskWord = bitsPerMaskWord / 2;
      constexpr uint64_t lowWordMask = 0x5555555555555555ULL;
      const uint32_t wordCount = (validCount + elementsPerMaskWord - 1) / elementsPerMaskWord;
      for (uint32_t i = 0; i < wordCount; ++i) {
        const uint64_t lowBits = maskLowWords.GetValue(i);
        const uint64_t highBits = maskHighWords.GetValue(i);
        uint64_t matched = lowBits & (highBits >> 1) & lowWordMask;

        const uint32_t first = i * elementsPerMaskWord;
        const uint32_t remaining = validCount - first;
        const uint32_t active = remaining < elementsPerMaskWord ? remaining : elementsPerMaskWord;
        if (active < elementsPerMaskWord) {
          const uint32_t activeBits = active * 2;
          matched &= ((1ULL << activeBits) - 1ULL) & lowWordMask;
        }
        count += static_cast<uint32_t>(GetBitCount<1>(matched));
      }
    }

    SetFlag<HardEvent::S_V>(scalarToVectorEvent_);
    WaitFlag<HardEvent::S_V>(scalarToVectorEvent_);

    return count;
  }
  __aicore__ inline void CopyOut(LocalTensor<IdType>& staging, uint64_t offset, uint32_t count) {

    const DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(IdType)), 0, 0, 0};
    SetFlag<HardEvent::S_MTE3>(scalarToMte3Event_);
    WaitFlag<HardEvent::S_MTE3>(scalarToMte3Event_);
    DataCopyPad(outputGm_[offset], staging, copyParams);
    SetFlag<HardEvent::MTE3_S>(mte3ToScalarEvent_);
    WaitFlag<HardEvent::MTE3_S>(mte3ToScalarEvent_);
  }
  uint64_t nnz_ = 0;
  uint64_t queryCount_ = 0;

  TPipe* pipe_ = nullptr;
  int32_t vectorToScalarEvent_;
  int32_t scalarToVectorEvent_;
  int32_t scalarToMte3Event_;
  int32_t mte3ToScalarEvent_;
  int32_t mte2ToVectorEvents_[BufferNum];

  const __gm__ IdType* queryRows_ = nullptr;

  GlobalTensor<IdType> cooRowsGm_;
  GlobalTensor<IdType> outputGm_;

  TBuf<TPosition::VECCALC> inputBuffer_;
  TBuf<TPosition::VECCALC> maskLowBuffer_;
  TBuf<TPosition::VECCALC> maskHighBuffer_;
  TBuf<TPosition::VECCALC> outputBuffer_;
};

}  // namespace

// =====================================================
// int32_t Kernel entry
// =====================================================

extern "C" __global__ __aicore__ void
coo_get_row_nnz_int32(GM_ADDR cooRows, GM_ADDR queryRows, GM_ADDR output, GM_ADDR tiling) {
  TPipe pipe;
  KernelCOOGetRowNNZ<int32_t> op;
  op.Init(cooRows, queryRows, output, tiling, &pipe);
  op.Process();
}

// =====================================================
// int64_t Kernel entry
// =====================================================

extern "C" __global__ __aicore__ void
coo_get_row_nnz_int64(GM_ADDR cooRows, GM_ADDR queryRows, GM_ADDR output, GM_ADDR tiling) {
  TPipe pipe;
  KernelCOOGetRowNNZ<int64_t> op;
  op.Init(cooRows, queryRows, output, tiling, &pipe);
  op.Process();
}
