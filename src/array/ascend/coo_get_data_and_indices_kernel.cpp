#include "kernel_operator.h"

#include <cstdint>

using namespace AscendC;

namespace {

constexpr uint32_t TileLength = 256;
constexpr uint32_t MaskBytes = TileLength / 8;
constexpr uint32_t MaskWordCount = MaskBytes / sizeof(uint64_t);
constexpr uint32_t DefaultMatchesPerQuery = 8;
constexpr uint32_t FlushElements = 256;
constexpr uint32_t MatchCountSlotElements = 32 / sizeof(uint64_t);
constexpr uint32_t BufferNum = 2;

// Keep the layout identical to coo_getData_and_indices.cc.
struct COOGetDataAndIndicesTiling {
  uint32_t nnz;
  uint32_t tileCount;
  uint32_t queryCount;
  uint32_t hasData;
};

template <typename IdType>
class KernelCOOQueryMatch {
 public:
  __aicore__ inline void Init(GM_ADDR cooRows, GM_ADDR cooCols, GM_ADDR queryRows, GM_ADDR queryCols, GM_ADDR cooData, GM_ADDR matchRows, GM_ADDR matchCounts, GM_ADDR matchCols, GM_ADDR matchData, GM_ADDR tiling_ptr, TPipe* pipe) {
    pipe_ = pipe;
    const __gm__ COOGetDataAndIndicesTiling* tiling = reinterpret_cast<const __gm__ COOGetDataAndIndicesTiling*>(tiling_ptr);
    nnz_ = tiling->nnz;
    queryCount_ = tiling->queryCount;
    hasData_ = tiling->hasData != 0;

    cooRowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(cooRows), nnz_);
    cooColsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(cooCols), nnz_);
    cooData_ = reinterpret_cast<const __gm__ IdType*>(cooData);
    queryRows_ = reinterpret_cast<const __gm__ IdType*>(queryRows);
    queryCols_ = reinterpret_cast<const __gm__ IdType*>(queryCols);
    const uint64_t matchElements = static_cast<uint64_t>(queryCount_) * DefaultMatchesPerQuery;
    matchRowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(matchRows), matchElements);
    matchColsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(matchCols), matchElements);
    matchDataGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(matchData), matchElements);
    matchCountsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(matchCounts),static_cast<uint64_t>(GetBlockNum()) * MatchCountSlotElements);

    pipe_->InitBuffer(rowsBuffer_, BufferNum * TileLength * sizeof(IdType));
    pipe_->InitBuffer(colsBuffer_, BufferNum * TileLength * sizeof(IdType));
    pipe_->InitBuffer(maskBuffer_, MaskBytes);
    pipe_->InitBuffer(rowMask_, MaskBytes);
    pipe_->InitBuffer(colMask_, MaskBytes);
    pipe_->InitBuffer(lowParts_, TileLength * sizeof(int32_t));
    pipe_->InitBuffer(highParts_, TileLength * sizeof(int32_t));
    pipe_->InitBuffer(matchCountBuffer_, 32);
    pipe_->InitBuffer(matchRowsBuffer_, BufferNum * FlushElements * sizeof(IdType));
    pipe_->InitBuffer(matchColsBuffer_, BufferNum * FlushElements * sizeof(IdType));
    pipe_->InitBuffer(matchDataBuffer_, BufferNum * FlushElements * sizeof(IdType));

    vectorToScalarEvent_ = static_cast<int32_t>(pipe_->AllocEventID<HardEvent::V_S>());
    scalarToVectorEvent_ = static_cast<int32_t>(pipe_->AllocEventID<HardEvent::S_V>());
    for (uint32_t slot = 0; slot < BufferNum; ++slot) {
      mte2ToVectorEvents_[slot] = static_cast<int32_t>(pipe_->AllocEventID<HardEvent::MTE2_V>());
      scalarToMte3Events_[slot] = static_cast<int32_t>(pipe_->AllocEventID<HardEvent::S_MTE3>());
      mte3ToScalarEvents_[slot] = static_cast<int32_t>(pipe_->AllocEventID<HardEvent::MTE3_S>());
    }
  }

  __aicore__ inline void Process() {
    const uint32_t blockIndex = GetBlockIdx();
    const uint32_t blockCount = GetBlockNum();
    const uint64_t queryBegin = static_cast<uint64_t>(queryCount_) * blockIndex / blockCount;
    const uint64_t queryEnd = static_cast<uint64_t>(queryCount_) * (blockIndex + 1) / blockCount;

    const uint64_t matchBase = queryBegin * DefaultMatchesPerQuery;
    const uint64_t matchCapacity = (queryEnd - queryBegin) * DefaultMatchesPerQuery;
    uint64_t matchCount = 0;  // 已经匹配的结果数量
    uint64_t flushCount = 0;  // 已经搬出的结果数量
    uint32_t outIndex = 0;  // 搬出块的写索引
    uint32_t matchSlot = 0; // 匹配所使用的 Slot
    uint32_t matchSlotsInFlight = 0;  // 匹配结果 Slot 是否正在搬出的位图
    const uint32_t fullTileCount = nnz_ / TileLength;
    const uint32_t tailCount = nnz_ % TileLength;

    for (uint64_t q = queryBegin; q < queryEnd; ++q) {
      const IdType queryRow = queryRows_[q];
      const IdType queryCol = queryCols_[q];

      if (fullTileCount != 0) {
        CopyIn(0, TileLength);
      } else if (tailCount != 0) {
        CopyIn(0, tailCount);
      }

      for (uint32_t tileIndex = 0; tileIndex < fullTileCount; ++tileIndex) {
        const uint32_t nextTile = tileIndex + 1;
        if (nextTile < fullTileCount) {
          CopyIn(nextTile, TileLength);
        } else if (nextTile == fullTileCount && tailCount != 0) {
          CopyIn(nextTile, tailCount);
        }
        ComputeTile<false>(tileIndex, TileLength, queryRow, queryCol, matchBase, matchCapacity, matchCount, flushCount, outIndex, matchSlot, matchSlotsInFlight);
      }

      if (tailCount != 0) {
        ComputeTile<true>(fullTileCount, tailCount, queryRow, queryCol, matchBase, matchCapacity, matchCount, flushCount, outIndex, matchSlot, matchSlotsInFlight);
      }
    }

    // 尾部处理没有搬出的 match
    if (outIndex != 0) {
      const uint32_t matchOffset = matchSlot * FlushElements * sizeof(IdType);
      LocalTensor<IdType> matchRows = matchRowsBuffer_.GetWithOffset<IdType>(FlushElements, matchOffset);
      LocalTensor<IdType> matchCols = matchColsBuffer_.GetWithOffset<IdType>(FlushElements, matchOffset);
      LocalTensor<IdType> matchData = matchDataBuffer_.GetWithOffset<IdType>(FlushElements, matchOffset);
      CopyOut(matchRows, matchCols, matchData, matchBase + flushCount, outIndex, matchSlot);
      matchSlotsInFlight |= 1U << matchSlot;
    }

    for (uint32_t slot = 0; slot < BufferNum; ++slot) {
      WaitForMatchSlot(slot, matchSlotsInFlight);
    }

    LocalTensor<uint64_t> countBuffer = matchCountBuffer_.Get<uint64_t>();
    countBuffer.SetValue(0, matchCount);
    for (uint32_t i = 1; i < MatchCountSlotElements; ++i) {
      countBuffer.SetValue(i, 0);
    }

    SetFlag<HardEvent::S_MTE3>(scalarToMte3Events_[0]);
    WaitFlag<HardEvent::S_MTE3>(scalarToMte3Events_[0]);
    DataCopy(matchCountsGm_[static_cast<uint64_t>(blockIndex) * MatchCountSlotElements], countBuffer, MatchCountSlotElements);
    SetFlag<HardEvent::MTE3_S>(mte3ToScalarEvents_[0]);
    WaitFlag<HardEvent::MTE3_S>(mte3ToScalarEvents_[0]);

    PipeBarrier<PIPE_ALL>();
    pipe_->ReleaseEventID<HardEvent::V_S>(vectorToScalarEvent_);
    pipe_->ReleaseEventID<HardEvent::S_V>(scalarToVectorEvent_);
    for (uint32_t slot = 0; slot < BufferNum; ++slot) {
      pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVectorEvents_[slot]);
      pipe_->ReleaseEventID<HardEvent::S_MTE3>(scalarToMte3Events_[slot]);
      pipe_->ReleaseEventID<HardEvent::MTE3_S>(mte3ToScalarEvents_[slot]);
    }
  }

 private:
  __aicore__ inline void CopyIn(uint32_t tileIndex, uint32_t validCount) {
    const uint32_t start = tileIndex * TileLength;
    const DataCopyExtParams copyParams{1, validCount * static_cast<uint32_t>(sizeof(IdType)),0, 0, 0};
    const DataCopyPadExtParams<IdType> padParams{false, 0, 0, static_cast<IdType>(0)};
    const uint32_t slot = tileIndex % BufferNum;
    const uint32_t inputOffset = slot * TileLength * sizeof(IdType);
    LocalTensor<IdType> rows = rowsBuffer_.GetWithOffset<IdType>(TileLength, inputOffset);
    LocalTensor<IdType> cols = colsBuffer_.GetWithOffset<IdType>(TileLength, inputOffset);
    DataCopyPad(rows, cooRowsGm_[start], copyParams, padParams);
    DataCopyPad(cols, cooColsGm_[start], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(mte2ToVectorEvents_[slot]);
  }

  __aicore__ inline void Compare(const LocalTensor<IdType>& rows, const LocalTensor<IdType>& cols, IdType queryRow, IdType queryCol, LocalTensor<uint8_t>& mask);

  template <bool IsTail>
  __aicore__ inline void ComputeTile(uint32_t tileIndex, uint32_t validCount, IdType queryRow, IdType queryCol, uint64_t matchBase, uint64_t matchCapacity, uint64_t& matchCount, uint64_t& flushCount, uint32_t& outIndex, uint32_t& matchSlot, uint32_t& matchSlotsInFlight) {
    const uint32_t slot = tileIndex % BufferNum;
    const uint32_t inputOffset = slot * TileLength * sizeof(IdType);
    LocalTensor<IdType> rows = rowsBuffer_.GetWithOffset<IdType>(TileLength, inputOffset);
    LocalTensor<IdType> cols = colsBuffer_.GetWithOffset<IdType>(TileLength, inputOffset);
    LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();

    WaitFlag<HardEvent::MTE2_V>(mte2ToVectorEvents_[slot]);
    Compare(rows, cols, queryRow, queryCol, mask);
    SetFlag<HardEvent::V_S>(vectorToScalarEvent_);
    WaitFlag<HardEvent::V_S>(vectorToScalarEvent_);

    const uint32_t start = tileIndex * TileLength;
    // 清除尾块比较结果
    if constexpr (IsTail) {
      ClearTailMask(validCount, mask);
    }

    LocalTensor<uint64_t> maskWords = mask.ReinterpretCast<uint64_t>();
    uint64_t anyBits = 0;
    for (uint32_t word = 0; word < MaskWordCount; ++word) {
      anyBits |= maskWords.GetValue(word);
    }

    if (anyBits != 0) {
      uint32_t matchOffset = matchSlot * FlushElements * sizeof(IdType);
      LocalTensor<IdType> matchRows = matchRowsBuffer_.GetWithOffset<IdType>(FlushElements, matchOffset);
      LocalTensor<IdType> matchCols = matchColsBuffer_.GetWithOffset<IdType>(FlushElements, matchOffset);
      LocalTensor<IdType> matchData = matchDataBuffer_.GetWithOffset<IdType>(FlushElements, matchOffset);
      for (uint32_t word = 0; word < MaskWordCount; ++word) {
        uint64_t bits = maskWords.GetValue(word);
        while (bits != 0) {
          const uint32_t bit = static_cast<uint32_t>(GetSFFValue<1>(bits));
          bits &= bits - 1;
          const uint32_t cooIndex = start + word * 64 + bit;
          if (matchCount < matchCapacity) {
            matchRows.SetValue(outIndex, queryRow);
            matchCols.SetValue(outIndex, queryCol);
            matchData.SetValue(outIndex, hasData_ ? cooData_[cooIndex] : static_cast<IdType>(cooIndex));
            ++outIndex;
            if (outIndex == FlushElements) {
              CopyOut(matchRows, matchCols, matchData, matchBase + flushCount, outIndex, matchSlot);
              matchSlotsInFlight |= 1U << matchSlot;
              flushCount += outIndex;
              outIndex = 0;
              matchSlot = (matchSlot + 1) % BufferNum;
              WaitForMatchSlot(matchSlot, matchSlotsInFlight);
              matchOffset = matchSlot * FlushElements * sizeof(IdType);
              matchRows = matchRowsBuffer_.GetWithOffset<IdType>(FlushElements, matchOffset);
              matchCols = matchColsBuffer_.GetWithOffset<IdType>(FlushElements, matchOffset);
              matchData = matchDataBuffer_.GetWithOffset<IdType>(FlushElements, matchOffset);
            }
          }
          ++matchCount;
        }
      }
    }
    SetFlag<HardEvent::S_V>(scalarToVectorEvent_);
    WaitFlag<HardEvent::S_V>(scalarToVectorEvent_);
  }

  __aicore__ inline void WaitForMatchSlot(
      uint32_t slot, uint32_t& slotsInFlight) {
    const uint32_t slotBit = 1U << slot;
    if ((slotsInFlight & slotBit) != 0) {
      WaitFlag<HardEvent::MTE3_S>(mte3ToScalarEvents_[slot]);
      slotsInFlight &= ~slotBit;
    }
  }

  __aicore__ inline void ClearTailMask(uint32_t validCount, LocalTensor<uint8_t>& mask) {
    uint32_t first_invalid_byte = validCount / 8;
    const uint32_t valid_bits = validCount % 8;
    // 修改部分有效的字节
    if (valid_bits != 0) {
      const uint8_t keep_mask = static_cast<uint8_t>((1U << valid_bits) - 1U);
      mask.SetValue(first_invalid_byte, static_cast<uint8_t>(mask.GetValue(first_invalid_byte) & keep_mask));
      ++first_invalid_byte;
    }
    // 修改全部无效的字节
    for (uint32_t byte = first_invalid_byte;byte < MaskBytes; ++byte) {
      mask.SetValue(byte, 0);
    }
  }

  __aicore__ inline void CopyOut(LocalTensor<IdType>& rows, LocalTensor<IdType>& cols, LocalTensor<IdType>& data, uint64_t outputIndex, uint32_t count, uint32_t slot) {
    SetFlag<HardEvent::S_MTE3>(scalarToMte3Events_[slot]);
    WaitFlag<HardEvent::S_MTE3>(scalarToMte3Events_[slot]);
    const DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(IdType)), 0, 0, 0};
    DataCopyPad(matchRowsGm_[outputIndex], rows, copyParams);
    DataCopyPad(matchColsGm_[outputIndex], cols, copyParams);
    DataCopyPad(matchDataGm_[outputIndex], data, copyParams);
    SetFlag<HardEvent::MTE3_S>(mte3ToScalarEvents_[slot]);
  }

  TPipe* pipe_ = nullptr;

  TBuf<TPosition::VECCALC> rowsBuffer_;
  TBuf<TPosition::VECCALC> colsBuffer_;
  TBuf<TPosition::VECCALC> maskBuffer_;
  TBuf<TPosition::VECCALC> rowMask_;
  TBuf<TPosition::VECCALC> colMask_;
  TBuf<TPosition::VECCALC> lowParts_;
  TBuf<TPosition::VECCALC> highParts_;
  TBuf<TPosition::VECCALC> matchCountBuffer_;
  TBuf<TPosition::VECCALC> matchRowsBuffer_;
  TBuf<TPosition::VECCALC> matchColsBuffer_;
  TBuf<TPosition::VECCALC> matchDataBuffer_;

  int32_t vectorToScalarEvent_;
  int32_t scalarToVectorEvent_;
  int32_t mte2ToVectorEvents_[BufferNum];
  int32_t scalarToMte3Events_[BufferNum];
  int32_t mte3ToScalarEvents_[BufferNum];

  GlobalTensor<IdType> cooRowsGm_;
  GlobalTensor<IdType> cooColsGm_;
  GlobalTensor<IdType> matchRowsGm_;
  GlobalTensor<IdType> matchColsGm_;
  GlobalTensor<IdType> matchDataGm_;
  GlobalTensor<uint64_t> matchCountsGm_;

  const __gm__ IdType* queryRows_ = nullptr;
  const __gm__ IdType* queryCols_ = nullptr;
  const __gm__ IdType* cooData_ = nullptr;

  uint32_t nnz_ = 0;
  uint32_t queryCount_ = 0;
  bool hasData_ = false;
};

template <>
__aicore__ inline void
KernelCOOQueryMatch<int32_t>::Compare(const LocalTensor<int32_t>& rows, const LocalTensor<int32_t>& cols, int32_t queryRow, int32_t queryCol, LocalTensor<uint8_t>& mask) {
  LocalTensor<uint8_t> rowMask = rowMask_.Get<uint8_t>();
  LocalTensor<uint8_t> colMask = colMask_.Get<uint8_t>();

  CompareScalar(rowMask, rows, queryRow, CMPMODE::EQ, TileLength);
  CompareScalar(colMask, cols, queryCol, CMPMODE::EQ, TileLength);
  PipeBarrier<PIPE_V>();

  And(mask.ReinterpretCast<uint16_t>(), rowMask.ReinterpretCast<uint16_t>(), colMask.ReinterpretCast<uint16_t>(), static_cast<int32_t>(MaskBytes / sizeof(uint16_t)));
  PipeBarrier<PIPE_V>();
}

template <>
__aicore__ inline void
KernelCOOQueryMatch<int64_t>::Compare(const LocalTensor<int64_t>& rows, const LocalTensor<int64_t>& cols, int64_t queryRow, int64_t queryCol, LocalTensor<uint8_t>& mask) {
  LocalTensor<uint8_t> rowMask = rowMask_.Get<uint8_t>();
  LocalTensor<uint8_t> colMask = colMask_.Get<uint8_t>();
  LocalTensor<int32_t> lowParts = lowParts_.Get<int32_t>();
  LocalTensor<int32_t> highParts = highParts_.Get<int32_t>();

  constexpr uint16_t splitRepeats = TileLength * sizeof(int64_t) / 256;
  const GatherMaskParams splitParams{1, splitRepeats, 8, 0};
  uint64_t reservedCount = 0;

  const uint64_t queryRowBits = static_cast<uint64_t>(queryRow);
  const int32_t queryRowLow = static_cast<int32_t>(queryRowBits & 0xffffffffULL);
  const int32_t queryRowHigh = static_cast<int32_t>(queryRowBits >> 32);

  GatherMask(lowParts, rows.ReinterpretCast<int32_t>(), static_cast<uint8_t>(1), false, 0, splitParams, reservedCount);
  GatherMask(highParts, rows.ReinterpretCast<int32_t>(), static_cast<uint8_t>(2), false, 0, splitParams, reservedCount);
  PipeBarrier<PIPE_V>();

  CompareScalar(rowMask, lowParts, queryRowLow, CMPMODE::EQ, TileLength);
  CompareScalar(colMask, highParts, queryRowHigh, CMPMODE::EQ, TileLength);
  PipeBarrier<PIPE_V>();

  And(rowMask.ReinterpretCast<uint16_t>(), rowMask.ReinterpretCast<uint16_t>(), colMask.ReinterpretCast<uint16_t>(), static_cast<int32_t>(MaskBytes / sizeof(uint16_t)));
  PipeBarrier<PIPE_V>();

  const uint64_t queryColBits = static_cast<uint64_t>(queryCol);
  const int32_t queryColLow = static_cast<int32_t>(queryColBits & 0xffffffffULL);
  const int32_t queryColHigh = static_cast<int32_t>(queryColBits >> 32);

  GatherMask(lowParts, cols.ReinterpretCast<int32_t>(), static_cast<uint8_t>(1), false, 0, splitParams, reservedCount);
  GatherMask(highParts, cols.ReinterpretCast<int32_t>(), static_cast<uint8_t>(2), false, 0, splitParams, reservedCount);
  PipeBarrier<PIPE_V>();

  CompareScalar(colMask, lowParts, queryColLow, CMPMODE::EQ, TileLength);
  CompareScalar(mask, highParts, queryColHigh, CMPMODE::EQ, TileLength);
  PipeBarrier<PIPE_V>();

  And(colMask.ReinterpretCast<uint16_t>(), colMask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(), static_cast<int32_t>(MaskBytes / sizeof(uint16_t)));
  PipeBarrier<PIPE_V>();

  And(mask.ReinterpretCast<uint16_t>(), rowMask.ReinterpretCast<uint16_t>(), colMask.ReinterpretCast<uint16_t>(), static_cast<int32_t>(MaskBytes / sizeof(uint16_t)));
  PipeBarrier<PIPE_V>();
}

template <typename IdType>
class KernelCOOQueryGather {
 public:
  __aicore__ inline void Init(GM_ADDR matchRows, GM_ADDR matchCols, GM_ADDR matchData, GM_ADDR output_offsets, GM_ADDR outRows, GM_ADDR outCols, GM_ADDR outData, GM_ADDR tiling_ptr, TPipe* pipe) {
    pipe_ = pipe;
    if (GetBlockIdx() != 0) return;

    const __gm__ COOGetDataAndIndicesTiling* tiling = reinterpret_cast<const __gm__ COOGetDataAndIndicesTiling*>(tiling_ptr);
    queryCount_ = tiling->queryCount;
    outputOffsets_ = reinterpret_cast<const __gm__ uint64_t*>(output_offsets);

    const uint32_t blockCount = GetBlockNum();
    const uint64_t total = outputOffsets_[blockCount];
    const uint64_t matchElements = static_cast<uint64_t>(queryCount_) * DefaultMatchesPerQuery;
    matchRowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(matchRows), matchElements);
    matchColsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(matchCols), matchElements);
    matchDataGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(matchData), matchElements);

    outRows_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(outRows), total);
    outCols_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(outCols), total);
    outData_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(outData), total);

    pipe_->InitBuffer(inRowsQueue_, BufferNum, FlushElements * sizeof(IdType));
    pipe_->InitBuffer(inColsQueue_, BufferNum, FlushElements * sizeof(IdType));
    pipe_->InitBuffer(inDataQueue_, BufferNum, FlushElements * sizeof(IdType));

    for (uint32_t slot = 0; slot < BufferNum; ++slot) {
      mte2ToMte3Events_[slot] = static_cast<int32_t>(pipe_->AllocEventID<HardEvent::MTE2_MTE3>());
      mte3ToMte2Events_[slot] = static_cast<int32_t>(pipe_->AllocEventID<HardEvent::MTE3_MTE2>());
    }
  }

  __aicore__ inline void Process() {
    if (GetBlockIdx() != 0) return;
    const uint32_t blockCount = GetBlockNum();
    uint32_t copyOutTileIndex = 0;

    for (uint32_t block = 0; block < blockCount; ++block) {
      const uint64_t queryBegin = static_cast<uint64_t>(queryCount_) * block / blockCount;
      uint64_t sourceIndex = queryBegin * DefaultMatchesPerQuery;
      uint64_t outputIndex = outputOffsets_[block];
      uint64_t remaining = outputOffsets_[block + 1] - outputIndex;

      while (remaining != 0) {
        const uint32_t count = static_cast<uint32_t>(remaining < FlushElements ? remaining : FlushElements);
        CopyOut(sourceIndex, outputIndex, count, copyOutTileIndex);
        sourceIndex += count;
        outputIndex += count;
        remaining -= count;
        ++copyOutTileIndex;
      }
    }

    const uint32_t liveSlots = copyOutTileIndex < BufferNum ? copyOutTileIndex : BufferNum;
    for (uint32_t slot = 0; slot < liveSlots; ++slot) {
      WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[slot]);
    }
    for (uint32_t slot = 0; slot < BufferNum; ++slot) {
      pipe_->ReleaseEventID<HardEvent::MTE2_MTE3>(mte2ToMte3Events_[slot]);
      pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[slot]);
    }
  }

 private:
  __aicore__ inline void CopyOut(uint64_t sourceIndex, uint64_t outputIndex, uint32_t count, uint32_t copyOutTileIndex) {
    const uint32_t slot = copyOutTileIndex % BufferNum;

    if (copyOutTileIndex >= BufferNum) {
      WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[slot]);
    }

    const uint32_t copyBytes = count * static_cast<uint32_t>(sizeof(IdType));
    const DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
    const DataCopyPadExtParams<IdType> padParams{false, 0, 0, static_cast<IdType>(0)};

    LocalTensor<IdType> rows = inRowsQueue_.AllocTensor<IdType>();
    LocalTensor<IdType> cols = inColsQueue_.AllocTensor<IdType>();
    LocalTensor<IdType> data = inDataQueue_.AllocTensor<IdType>();
    DataCopyPad(rows, matchRowsGm_[sourceIndex], copyParams, padParams);
    DataCopyPad(cols, matchColsGm_[sourceIndex], copyParams, padParams);
    DataCopyPad(data, matchDataGm_[sourceIndex], copyParams, padParams);
    inRowsQueue_.EnQue(rows);
    inColsQueue_.EnQue(cols);
    inDataQueue_.EnQue(data);

    rows = inRowsQueue_.DeQue<IdType>();
    cols = inColsQueue_.DeQue<IdType>();
    data = inDataQueue_.DeQue<IdType>();
    SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Events_[slot]);
    WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Events_[slot]);

    DataCopyPad(outRows_gm_[outputIndex], rows, copyParams);
    DataCopyPad(outCols_gm_[outputIndex], cols, copyParams);
    DataCopyPad(outData_gm_[outputIndex], data, copyParams);
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[slot]);

    inRowsQueue_.FreeTensor(rows);
    inColsQueue_.FreeTensor(cols);
    inDataQueue_.FreeTensor(data);
  }

  TPipe* pipe_ = nullptr;

  TQue<TPosition::VECIN, BufferNum> inRowsQueue_;
  TQue<TPosition::VECIN, BufferNum> inColsQueue_;
  TQue<TPosition::VECIN, BufferNum> inDataQueue_;

  GlobalTensor<IdType> matchRowsGm_;
  GlobalTensor<IdType> matchColsGm_;
  GlobalTensor<IdType> matchDataGm_;
  GlobalTensor<IdType> outRows_gm_;
  GlobalTensor<IdType> outCols_gm_;
  GlobalTensor<IdType> outData_gm_;

  const __gm__ uint64_t* outputOffsets_ = nullptr;
  int32_t mte2ToMte3Events_[BufferNum];
  int32_t mte3ToMte2Events_[BufferNum];
  uint32_t queryCount_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void
coo_getData_and_indices(uint32_t operation, GM_ADDR cooRows, GM_ADDR cooCols, GM_ADDR queryRows, GM_ADDR queryCols, GM_ADDR cooData, GM_ADDR mask, GM_ADDR offsets, GM_ADDR outRows, GM_ADDR outCols, GM_ADDR outData, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  if (operation == 0) {
    KernelCOOQueryMatch<int32_t> op;
    op.Init(cooRows, cooCols, queryRows, queryCols, cooData, mask, offsets, outRows, outCols, tiling, &pipe);
    op.Process();
  } else if (operation == 1) {
    KernelCOOQueryMatch<int64_t> op;
    op.Init(cooRows, cooCols, queryRows, queryCols, cooData, mask, offsets, outRows, outCols, tiling, &pipe);
    op.Process();
  } else if (operation == 2) {
    KernelCOOQueryGather<int32_t> op;
    op.Init(cooRows, cooCols, cooData, offsets, outRows, outCols, outData, tiling, &pipe);
    op.Process();
  } else if (operation == 3) {
    KernelCOOQueryGather<int64_t> op;
    op.Init(cooRows, cooCols, cooData, offsets, outRows, outCols, outData, tiling, &pipe);
    op.Process();
  }
}
