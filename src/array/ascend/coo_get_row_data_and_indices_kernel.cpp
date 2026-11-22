#include "kernel_operator.h"
#include "coo_get_row_data_and_indices_tiling.h"

namespace {

template <typename IdType>
class CountKernel {
 public:
  __aicore__ inline void Init(GM_ADDR rows, GM_ADDR counts, GM_ADDR tiling) {
    const __gm__ CooGetRowTiling<IdType>* params = reinterpret_cast<const __gm__ CooGetRowTiling<IdType>*>(tiling);
    nnz_ = params->nnz;
    targetRow_ = params->targetRow;
    blockIdx_ = AscendC::GetBlockIdx();
    blockNum_ = AscendC::GetBlockNum();
    const uint32_t perBlock = nnz_ / blockNum_;
    const uint32_t remainder = nnz_ % blockNum_;
    const uint32_t extra = blockIdx_ < remainder ? blockIdx_ : remainder;
    begin_ = blockIdx_ * perBlock + extra;
    end_ = begin_ + perBlock + (blockIdx_ < remainder ? 1U : 0U);

    rowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(rows), nnz_);
    countsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(counts));
    pipe_.InitBuffer(rowsQueue_, 1, kCooGetRowTile * sizeof(IdType));
    pipe_.InitBuffer(countBuffer_, kCooGetRowCountWords * sizeof(uint32_t));
  }

  __aicore__ inline void Process() {
    uint32_t count = 0;
    uint32_t base = begin_;
    while (base < end_) {
      const uint32_t remaining = end_ - base;
      const uint32_t length = remaining < kCooGetRowTile ? remaining : kCooGetRowTile;
      const uint32_t bytes = length * static_cast<uint32_t>(sizeof(IdType));

      AscendC::LocalTensor<IdType> local = rowsQueue_.AllocTensor<IdType>();
      AscendC::DataCopyExtParams copy{1, bytes, 0, 0, 0};
      AscendC::DataCopyPadExtParams<IdType> pad{false, 0, 0, 0};
      AscendC::DataCopyPad(local, rowsGm_[base], copy, pad);
      rowsQueue_.EnQue(local);
      local = rowsQueue_.DeQue<IdType>();

      for (uint32_t i = 0; i < length; ++i) {
        if (local.GetValue(i) == targetRow_) {
          ++count;
        }
      }
      rowsQueue_.FreeTensor(local);
      base += length;
    }

    AscendC::LocalTensor<uint32_t> count_local = countBuffer_.Get<uint32_t>();
    for (uint32_t i = 0; i < kCooGetRowCountWords; ++i) {
      count_local.SetValue(i, 0);
    }
    count_local.SetValue(0, count);
    AscendC::DataCopyExtParams copy{1, kCooGetRowCountWords * sizeof(uint32_t), 0, 0, 0};
    AscendC::DataCopyPad(countsGm_[blockIdx_ * kCooGetRowCountWords], count_local, copy);
  }

 private:
  AscendC::TPipe pipe_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> rowsQueue_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> countBuffer_;
  AscendC::GlobalTensor<IdType> rowsGm_;
  AscendC::GlobalTensor<uint32_t> countsGm_;
  uint32_t nnz_ = 0;
  IdType targetRow_ = 0;
  uint32_t blockIdx_ = 0;
  uint32_t blockNum_ = 1;
  uint32_t begin_ = 0;
  uint32_t end_ = 0;
};

template <typename IdType>
class ScatterKernel {
 public:
  __aicore__ inline void Init(GM_ADDR rows, GM_ADDR cols, GM_ADDR data, GM_ADDR offsets, GM_ADDR out_data, GM_ADDR out_cols, GM_ADDR tiling) {
    const __gm__ CooGetRowTiling<IdType>* params = reinterpret_cast<const __gm__ CooGetRowTiling<IdType>*>(tiling);
    nnz_ = params->nnz;
    targetRow_ = params->targetRow;
    hasData_ = params->hasData != 0;
    blockIdx_ = AscendC::GetBlockIdx();
    blockNum_ = AscendC::GetBlockNum();
    const uint32_t perBlock = nnz_ / blockNum_;
    const uint32_t remainder = nnz_ % blockNum_;
    const uint32_t extra = blockIdx_ < remainder ? blockIdx_ : remainder;
    begin_ = blockIdx_ * perBlock + extra;
    end_ = begin_ + perBlock + (blockIdx_ < remainder ? 1U : 0U);

    rowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(rows), nnz_);
    colsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(cols), nnz_);
    if (hasData_) {
      data_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(data), nnz_);
    }
    offsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(offsets));
    outDataGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(out_data));
    outColsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(out_cols));

    pipe_.InitBuffer(rowsQueue_, 1, kCooGetRowTile * sizeof(IdType));
    pipe_.InitBuffer(colsQueue_, 1, kCooGetRowTile * sizeof(IdType));
    pipe_.InitBuffer(dataQueue_, 1, kCooGetRowTile * sizeof(IdType));
    pipe_.InitBuffer(outDataQueue_, 1, kCooGetRowTile * sizeof(IdType));
    pipe_.InitBuffer(outColsQueue_, 1, kCooGetRowTile * sizeof(IdType));
  }

  __aicore__ inline void Process() {
    uint32_t write_pos = offsetsGm_.GetValue(blockIdx_);

    uint32_t base = begin_;
    while (base < end_) {
      const uint32_t remaining = end_ - base;
      const uint32_t length = remaining < kCooGetRowTile ? remaining : kCooGetRowTile;
      const uint32_t bytes = length * static_cast<uint32_t>(sizeof(IdType));
      AscendC::DataCopyExtParams copy{1, bytes, 0, 0, 0};
      AscendC::DataCopyPadExtParams<IdType> pad{false, 0, 0, 0};

      AscendC::LocalTensor<IdType> rowsLocal = rowsQueue_.AllocTensor<IdType>();
      AscendC::DataCopyPad(rowsLocal, rowsGm_[base], copy, pad);
      rowsQueue_.EnQue(rowsLocal);
      rowsLocal = rowsQueue_.DeQue<IdType>();

      AscendC::LocalTensor<IdType> rowsLocal = colsQueue_.AllocTensor<IdType>();
      AscendC::DataCopyPad(rowsLocal, colsGm_[base], copy, pad);
      colsQueue_.EnQue(rowsLocal);
      rowsLocal = colsQueue_.DeQue<IdType>();

      AscendC::LocalTensor<IdType> data_local;
      if (hasData_) {
        data_local = dataQueue_.AllocTensor<IdType>();
        AscendC::DataCopyPad(data_local, data_gm_[base], copy, pad);
        dataQueue_.EnQue(data_local);
        data_local = dataQueue_.DeQue<IdType>();
      }

      AscendC::LocalTensor<IdType> resultData = outDataQueue_.AllocTensor<IdType>();
      AscendC::LocalTensor<IdType> resultCols = outColsQueue_.AllocTensor<IdType>();
      uint32_t resultIndex = 0;
      for (uint32_t i = 0; i < length; ++i) {
        if (rowsLocal.GetValue(i) == targetRow_) {
          resultCols.SetValue(resultIndex, rowsLocal.GetValue(i));
          resultData.SetValue(resultIndex, hasData_ ? data_local.GetValue(i) : static_cast<IdType>(base + i));
          ++resultIndex;
        }
      }

      outDataQueue_.EnQue(resultData);
      outColsQueue_.EnQue(resultCols);
      resultData = outDataQueue_.DeQue<IdType>();
      resultCols = outColsQueue_.DeQue<IdType>();
      if (resultIndex != 0) {
        const uint32_t outputBytes = resultIndex * static_cast<uint32_t>(sizeof(IdType));
        AscendC::DataCopyExtParams outputCopy{1, outputBytes, 0, 0, 0};
        AscendC::DataCopyPad(outDataGm_[write_pos], resultData, outputCopy);
        AscendC::DataCopyPad(outColsGm_[write_pos], resultCols, outputCopy);
        write_pos += resultIndex;
      }

      rowsQueue_.FreeTensor(rowsLocal);
      colsQueue_.FreeTensor(rowsLocal);
      if (hasData_) {
        dataQueue_.FreeTensor(data_local);
      }
      outDataQueue_.FreeTensor(resultData);
      outColsQueue_.FreeTensor(resultCols);
      base += length;
    }
  }

 private:
  AscendC::TPipe pipe_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> rowsQueue_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> colsQueue_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> dataQueue_;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outDataQueue_;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outColsQueue_;
  AscendC::GlobalTensor<IdType> rowsGm_;
  AscendC::GlobalTensor<IdType> colsGm_;
  AscendC::GlobalTensor<IdType> data_gm_;
  AscendC::GlobalTensor<uint32_t> offsetsGm_;
  AscendC::GlobalTensor<IdType> outDataGm_;
  AscendC::GlobalTensor<IdType> outColsGm_;
  uint32_t nnz_ = 0;
  IdType targetRow_ = 0;
  bool hasData_ = false;
  uint32_t blockIdx_ = 0;
  uint32_t blockNum_ = 1;
  uint32_t begin_ = 0;
  uint32_t end_ = 0;
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
