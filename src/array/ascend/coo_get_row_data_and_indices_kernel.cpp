#include "kernel_operator.h"
#include "coo_get_row_data_and_indices_tiling.h"

namespace {

using AscendC::HardEvent;

constexpr uint32_t kMaskBytes = kCooGetRowTile / 8;
constexpr uint32_t kInt64MaskBytes = 2 * kMaskBytes;

// 一个 ID 占几个 32 位比较单元
template <typename IdType>
constexpr uint32_t kLanesPerId = sizeof(IdType) / sizeof(int32_t);

template <typename IdType>
constexpr uint32_t kCompareBytes = kMaskBytes;
template <>
constexpr uint32_t kCompareBytes<int64_t> = 2 * kInt64MaskBytes;

__aicore__ inline void Compare(const AscendC::LocalTensor<int32_t>& rows, int32_t target, uint32_t length, AscendC::LocalTensor<uint8_t> mask, AscendC::TEventID vectorToScalarEvent) {
  const uint32_t wordCount = (length + 63) / 64;
  AscendC::CompareScalar(mask, rows, target, AscendC::CMPMODE::EQ, wordCount * 64);
  AscendC::SetFlag<HardEvent::V_S>(vectorToScalarEvent);
  AscendC::WaitFlag<HardEvent::V_S>(vectorToScalarEvent);
  const uint32_t tailBits = length % 64;
  if (tailBits != 0) {
    auto words = mask.ReinterpretCast<uint64_t>();
    words.SetValue(wordCount - 1, words.GetValue(wordCount - 1) & ((1ULL << tailBits) - 1));
  }
}

__aicore__ inline void Compare(const AscendC::LocalTensor<int64_t>& rows, int64_t target, uint32_t length, AscendC::LocalTensor<uint8_t> mask, AscendC::TEventID vectorToScalarEvent) {
  auto highMask = mask[kInt64MaskBytes];
  // 比较单元
  auto lanes = rows.ReinterpretCast<int32_t>();
  const uint32_t laneCount = length * kLanesPerId<int64_t>;
  const uint32_t wordCount = (laneCount + 63) / 64;
  const uint64_t bits = static_cast<uint64_t>(target);
  AscendC::CompareScalar(mask, lanes, static_cast<int32_t>(bits), AscendC::CMPMODE::EQ, wordCount * 64);
  AscendC::CompareScalar(highMask, lanes, static_cast<int32_t>(bits >> 32), AscendC::CMPMODE::EQ, wordCount * 64);
  AscendC::SetFlag<HardEvent::V_S>(vectorToScalarEvent);
  AscendC::WaitFlag<HardEvent::V_S>(vectorToScalarEvent);

  auto lowWords = mask.ReinterpretCast<uint64_t>();
  auto highWords = highMask.ReinterpretCast<uint64_t>();
  constexpr uint64_t evenLanes = 0x5555555555555555ULL;
  for (uint32_t word = 0; word < wordCount; ++word) {
    uint64_t matched = lowWords.GetValue(word) & (highWords.GetValue(word) >> 1) & evenLanes;
    const uint32_t validLanes = laneCount - word * 64;
    if (validLanes < 64) matched &= (1ULL << validLanes) - 1;
    lowWords.SetValue(word, matched);
  }
}

template <typename IdType>
class CountKernel {
 public:
  __aicore__ inline void Init(GM_ADDR rows, GM_ADDR counts, GM_ADDR tiling) {
    const auto* params = reinterpret_cast<const __gm__ CooGetRowTiling<IdType>*>(tiling);
    const uint32_t nnz = params->nnz;
    targetRow_ = params->targetRow;
    blockIdx_ = AscendC::GetBlockIdx();
    const uint32_t blockNum = AscendC::GetBlockNum();
    const uint32_t perBlock = nnz / blockNum;
    const uint32_t remainder = nnz % blockNum;
    const uint32_t extra = blockIdx_ < remainder ? blockIdx_ : remainder;
    begin_ = blockIdx_ * perBlock + extra;
    end_ = begin_ + perBlock + (blockIdx_ < remainder ? 1U : 0U);

    rowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(rows), nnz);
    countsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(counts));
    pipe_.InitBuffer(rowsQueue_, 2, kCooGetRowTile * sizeof(IdType));
    pipe_.InitBuffer(countBuffer_, kCooGetRowCountWords * sizeof(uint32_t));
    pipe_.InitBuffer(compareBuffer_, kCompareBytes<IdType>);
    vectorToScalarEvent_ = pipe_.AllocEventID<HardEvent::V_S>();
    scalarToVectorEvent_ = pipe_.AllocEventID<HardEvent::S_V>();
    scalarToMte3Event_ = pipe_.AllocEventID<HardEvent::S_MTE3>();
  }

  __aicore__ inline void Process() {
    uint32_t count = 0;
    uint32_t base = begin_;
    if (base < end_) CopyIn(base);
    while (base < end_) {
      const uint32_t remaining = end_ - base;
      const uint32_t length = remaining < kCooGetRowTile ? remaining : kCooGetRowTile;
      const uint32_t next = base + length;
      if (next < end_) CopyIn(next);

      auto rows = rowsQueue_.DeQue<IdType>();
      auto mask = compareBuffer_.Get<uint8_t>();
      Compare(rows, targetRow_, length, mask, vectorToScalarEvent_);
      auto maskWords = mask.ReinterpretCast<uint64_t>();
      const uint32_t wordCount = (length * kLanesPerId<IdType> + 63) / 64;
      for (uint32_t word = 0; word < wordCount; ++word) {
        count += AscendC::GetBitCount<1>(maskWords.GetValue(word));
      }
      AscendC::SetFlag<HardEvent::S_V>(scalarToVectorEvent_);
      AscendC::WaitFlag<HardEvent::S_V>(scalarToVectorEvent_);
      rowsQueue_.FreeTensor(rows);
      base = next;
    }

    auto countLocal = countBuffer_.Get<uint32_t>();
    countLocal.SetValue(0, count);
    AscendC::SetFlag<HardEvent::S_MTE3>(scalarToMte3Event_);
    AscendC::WaitFlag<HardEvent::S_MTE3>(scalarToMte3Event_);
    const AscendC::DataCopyExtParams copy{1, sizeof(uint32_t), 0, 0, 0};
    AscendC::DataCopyPad(countsGm_[blockIdx_ * kCooGetRowCountWords], countLocal, copy);
    pipe_.ReleaseEventID<HardEvent::V_S>(vectorToScalarEvent_);
    pipe_.ReleaseEventID<HardEvent::S_V>(scalarToVectorEvent_);
    pipe_.ReleaseEventID<HardEvent::S_MTE3>(scalarToMte3Event_);
  }

 private:
  __aicore__ inline void CopyIn(uint32_t base) {
    const uint32_t remaining = end_ - base;
    const uint32_t length = remaining < kCooGetRowTile ? remaining : kCooGetRowTile;
    const AscendC::DataCopyExtParams copy{1, length * static_cast<uint32_t>(sizeof(IdType)), 0, 0, 0};
    const AscendC::DataCopyPadExtParams<IdType> pad{false, 0, 0, 0};
    auto rows = rowsQueue_.AllocTensor<IdType>();
    AscendC::DataCopyPad(rows, rowsGm_[base], copy, pad);
    rowsQueue_.EnQue(rows);
  }

  AscendC::TPipe pipe_;
  AscendC::TEventID vectorToScalarEvent_, scalarToVectorEvent_, scalarToMte3Event_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 2> rowsQueue_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> countBuffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> compareBuffer_;
  AscendC::GlobalTensor<IdType> rowsGm_;
  AscendC::GlobalTensor<uint32_t> countsGm_;
  IdType targetRow_ = 0;
  uint32_t blockIdx_ = 0;
  uint32_t begin_ = 0;
  uint32_t end_ = 0;
};

template <typename IdType>
class ScatterKernel {
 public:
  __aicore__ inline void Init(GM_ADDR rows, GM_ADDR cols, GM_ADDR data, GM_ADDR offsets, GM_ADDR outData, GM_ADDR outCols, GM_ADDR tiling) {
    const auto* params = reinterpret_cast<const __gm__ CooGetRowTiling<IdType>*>(tiling);
    const uint32_t nnz = params->nnz;
    targetRow_ = params->targetRow;
    hasData_ = params->hasData != 0;
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    const uint32_t blockNum = AscendC::GetBlockNum();
    const uint32_t perBlock = nnz / blockNum;
    const uint32_t remainder = nnz % blockNum;
    const uint32_t extra = blockIdx < remainder ? blockIdx : remainder;
    begin_ = blockIdx * perBlock + extra;
    end_ = begin_ + perBlock + (blockIdx < remainder ? 1U : 0U);

    AscendC::GlobalTensor<uint32_t> startOffsets;
    startOffsets.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(offsets));
    writePos_ = startOffsets.GetValue(blockIdx);
    writeEnd_ = startOffsets.GetValue(blockIdx + 1);

    rowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(rows), nnz);
    colsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(cols), nnz);
    if (hasData_) {
      dataGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(data), nnz);
    }
    outDataGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(outData));
    outColsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(outCols));

    pipe_.InitBuffer(inputs_, 2, (hasData_ ? 3 : 2) * kCooGetRowTile * sizeof(IdType));
    pipe_.InitBuffer(outputs_, 2, 2 * kCooGetRowTile * sizeof(IdType));
    pipe_.InitBuffer(compareBuffer_, kCompareBytes<IdType>);
    vectorToScalarEvent_ = pipe_.AllocEventID<HardEvent::V_S>();
    scalarToVectorEvent_ = pipe_.AllocEventID<HardEvent::S_V>();
  }

  __aicore__ inline void Process() {
    if (writePos_ == writeEnd_) return;
    uint32_t base = begin_;
    const uint32_t remaining = end_ - base;
    uint32_t length = remaining < kCooGetRowTile ? remaining : kCooGetRowTile;
    CopyIn(base, length);
    auto output = outputs_.AllocTensor<IdType>();
    uint32_t outIndex = 0;
    while (base < end_) {
      const uint32_t next = base + length;
      const uint32_t nextRemaining = end_ - next;
      const uint32_t nextLength = nextRemaining < kCooGetRowTile ? nextRemaining : kCooGetRowTile;
      if (next < end_) CopyIn(next, nextLength);
      ComputeAndWrite(base, length, output, outIndex);
      if (next == end_) break;
      if (outIndex == writeEnd_ - writePos_) {
        auto prefetched = inputs_.DeQue<IdType>();
        inputs_.FreeTensor(prefetched);
        break;
      }
      if (writeEnd_ - writePos_ > kCooGetRowTile && outIndex + nextLength > kCooGetRowTile) {
        CopyOut(output, outIndex);
        output = outputs_.AllocTensor<IdType>();
        outIndex = 0;
      }
      base = next;
      length = nextLength;
    }
    CopyOut(output, outIndex);
    pipe_.ReleaseEventID<HardEvent::V_S>(vectorToScalarEvent_);
    pipe_.ReleaseEventID<HardEvent::S_V>(scalarToVectorEvent_);
  }

 private:
  __aicore__ inline void CopyIn(uint32_t base, uint32_t length) {
    auto input = inputs_.AllocTensor<IdType>();
    const AscendC::DataCopyExtParams copy{1, length * static_cast<uint32_t>(sizeof(IdType)), 0, 0, 0};
    const AscendC::DataCopyPadExtParams<IdType> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(input, rowsGm_[base], copy, pad);
    AscendC::DataCopyPad(input[kCooGetRowTile], colsGm_[base], copy, pad);
    if (hasData_) {
      AscendC::DataCopyPad(input[2 * kCooGetRowTile], dataGm_[base], copy, pad);
    }
    inputs_.EnQue(input);
  }

  __aicore__ inline void ComputeAndWrite(uint32_t base, uint32_t length, AscendC::LocalTensor<IdType> output, uint32_t& outIndex) {
    auto input = inputs_.DeQue<IdType>();
    auto rowMask = compareBuffer_.Get<uint8_t>();
    Compare(input, targetRow_, length, rowMask, vectorToScalarEvent_);
    auto maskWords = rowMask.ReinterpretCast<uint64_t>();
    auto cols = input[kCooGetRowTile];
    auto outData = output[kCooGetRowTile];
    const uint32_t wordCount = (length * kLanesPerId<IdType> + 63) / 64;
    for (uint32_t word = 0; word < wordCount; ++word) {
      uint64_t bits = maskWords.GetValue(word);
      while (bits != 0) {
        const uint32_t bit = static_cast<uint32_t>(AscendC::GetSFFValue<1>(bits));
        bits &= bits - 1;
        const uint32_t index = (word * 64 + bit) / kLanesPerId<IdType>;
        output.SetValue(outIndex, cols.GetValue(index));
        outData.SetValue(outIndex, hasData_ ? input.GetValue(2 * kCooGetRowTile + index) : static_cast<IdType>(base + index));
        ++outIndex;
      }
    }

    AscendC::SetFlag<HardEvent::S_V>(scalarToVectorEvent_);
    AscendC::WaitFlag<HardEvent::S_V>(scalarToVectorEvent_);
    inputs_.FreeTensor(input);
  }

  __aicore__ inline void CopyOut(AscendC::LocalTensor<IdType> output, uint32_t count) {
    outputs_.EnQue(output);
    output = outputs_.DeQue<IdType>();
    if (count != 0) {
      const AscendC::DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(IdType)), 0, 0, 0};
      AscendC::DataCopyPad(outColsGm_[writePos_], output, copy);
      AscendC::DataCopyPad(outDataGm_[writePos_], output[kCooGetRowTile], copy);
      writePos_ += count;
    }
    outputs_.FreeTensor(output);
  }

  AscendC::TPipe pipe_;
  AscendC::TEventID vectorToScalarEvent_, scalarToVectorEvent_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 2> inputs_;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outputs_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> compareBuffer_;
  AscendC::GlobalTensor<IdType> rowsGm_, colsGm_, dataGm_;
  AscendC::GlobalTensor<IdType> outDataGm_, outColsGm_;
  IdType targetRow_ = 0;
  bool hasData_ = false;
  uint32_t begin_ = 0;
  uint32_t end_ = 0;
  uint32_t writePos_ = 0;
  uint32_t writeEnd_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void coo_get_row_count_i32(GM_ADDR rows, GM_ADDR counts, GM_ADDR tiling) {
  CountKernel<int32_t> op;
  op.Init(rows, counts, tiling);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_get_row_scatter_i32(GM_ADDR rows, GM_ADDR cols, GM_ADDR data, GM_ADDR offsets, GM_ADDR out_data, GM_ADDR out_cols, GM_ADDR tiling) {
  ScatterKernel<int32_t> op;
  op.Init(rows, cols, data, offsets, out_data, out_cols, tiling);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_get_row_count_i64(GM_ADDR rows, GM_ADDR counts, GM_ADDR tiling) {
  CountKernel<int64_t> op;
  op.Init(rows, counts, tiling);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_get_row_scatter_i64(GM_ADDR rows, GM_ADDR cols, GM_ADDR data, GM_ADDR offsets, GM_ADDR out_data, GM_ADDR out_cols, GM_ADDR tiling) {
  ScatterKernel<int64_t> op;
  op.Init(rows, cols, data, offsets, out_data, out_cols, tiling);
  op.Process();
}
