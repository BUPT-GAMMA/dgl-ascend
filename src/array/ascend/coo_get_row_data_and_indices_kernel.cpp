/**
 * Copyright (c) 2024 by Contributors
 * @file coo_get_row_data_and_indices_kernel.cpp
 * @brief Stable Count/Scatter kernels for selecting one COO row.
 */
#include "kernel_operator.h"

#include "coo_get_row_data_and_indices_tiling.h"

namespace {

__aicore__ inline void GetBlockRange(
    uint32_t nnz, uint32_t block_idx, uint32_t block_num, uint32_t* begin,
    uint32_t* end) {
  const uint32_t per_block = nnz / block_num;
  const uint32_t remainder = nnz % block_num;
  const uint32_t extra_before = block_idx < remainder ? block_idx : remainder;
  *begin = block_idx * per_block + extra_before;
  *end = *begin + per_block + (block_idx < remainder ? 1U : 0U);
}

template <typename IdType>
class CountKernel {
 public:
  __aicore__ inline void Init(GM_ADDR rows, GM_ADDR count_slabs,
                              GM_ADDR tiling) {
    const __gm__ CooGetRowTiling<IdType>* params =
        reinterpret_cast<const __gm__ CooGetRowTiling<IdType>*>(tiling);
    nnz_ = params->nnz;
    target_row_ = params->target_row;
    block_idx_ = AscendC::GetBlockIdx();
    block_num_ = AscendC::GetBlockNum();
    GetBlockRange(nnz_, block_idx_, block_num_, &begin_, &end_);

    rows_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(rows), nnz_);
    counts_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(count_slabs));
    pipe_.InitBuffer(rows_queue_, 1,
                     kCooGetRowChunkElements * sizeof(IdType));
    pipe_.InitBuffer(count_buffer_,
                     kCooGetRowCountStrideWords * sizeof(uint32_t));
  }

  __aicore__ inline void Process() {
    uint32_t count = 0;
    uint32_t base = begin_;
    while (base < end_) {
      const uint32_t remaining = end_ - base;
      const uint32_t length =
          remaining < kCooGetRowChunkElements ? remaining
                                               : kCooGetRowChunkElements;
      const uint32_t bytes =
          length * static_cast<uint32_t>(sizeof(IdType));

      AscendC::LocalTensor<IdType> local =
          rows_queue_.AllocTensor<IdType>();
      AscendC::DataCopyExtParams copy{1, bytes, 0, 0, 0};
      AscendC::DataCopyPadExtParams<IdType> pad{false, 0, 0, 0};
      AscendC::DataCopyPad(local, rows_gm_[base], copy, pad);
      rows_queue_.EnQue(local);
      local = rows_queue_.DeQue<IdType>();

      for (uint32_t i = 0; i < length; ++i) {
        if (local.GetValue(i) == target_row_) {
          ++count;
        }
      }
      rows_queue_.FreeTensor(local);
      base += length;
    }

    AscendC::LocalTensor<uint32_t> count_local =
        count_buffer_.Get<uint32_t>();
    for (uint32_t i = 0; i < kCooGetRowCountStrideWords; ++i) {
      count_local.SetValue(i, 0);
    }
    count_local.SetValue(0, count);
    AscendC::DataCopyExtParams copy{
        1, kCooGetRowCountStrideWords * sizeof(uint32_t), 0, 0, 0};
    AscendC::DataCopyPad(
        counts_gm_[block_idx_ * kCooGetRowCountStrideWords], count_local,
        copy);
  }

 private:
  AscendC::TPipe pipe_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> rows_queue_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> count_buffer_;
  AscendC::GlobalTensor<IdType> rows_gm_;
  AscendC::GlobalTensor<uint32_t> counts_gm_;
  uint32_t nnz_ = 0;
  IdType target_row_ = 0;
  uint32_t block_idx_ = 0;
  uint32_t block_num_ = 1;
  uint32_t begin_ = 0;
  uint32_t end_ = 0;
};

template <typename IdType>
class ScatterKernel {
 public:
  __aicore__ inline void Init(
      GM_ADDR rows, GM_ADDR cols, GM_ADDR data, GM_ADDR offsets,
      GM_ADDR out_data, GM_ADDR out_cols, GM_ADDR tiling) {
    const __gm__ CooGetRowTiling<IdType>* params =
        reinterpret_cast<const __gm__ CooGetRowTiling<IdType>*>(tiling);
    nnz_ = params->nnz;
    target_row_ = params->target_row;
    has_data_ = params->has_data != 0;
    block_idx_ = AscendC::GetBlockIdx();
    block_num_ = AscendC::GetBlockNum();
    GetBlockRange(nnz_, block_idx_, block_num_, &begin_, &end_);

    rows_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(rows), nnz_);
    cols_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(cols), nnz_);
    if (has_data_) {
      data_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(data), nnz_);
    }
    offsets_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(offsets));
    out_data_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(out_data));
    out_cols_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(out_cols));

    pipe_.InitBuffer(rows_queue_, 1,
                     kCooGetRowChunkElements * sizeof(IdType));
    pipe_.InitBuffer(cols_queue_, 1,
                     kCooGetRowChunkElements * sizeof(IdType));
    pipe_.InitBuffer(data_queue_, 1,
                     kCooGetRowChunkElements * sizeof(IdType));
    pipe_.InitBuffer(out_data_queue_, 1,
                     kCooGetRowChunkElements * sizeof(IdType));
    pipe_.InitBuffer(out_cols_queue_, 1,
                     kCooGetRowChunkElements * sizeof(IdType));
    pipe_.InitBuffer(offset_queue_, 1, 32);
  }

  __aicore__ inline void Process() {
    AscendC::LocalTensor<uint32_t> offset_local =
        offset_queue_.AllocTensor<uint32_t>();
    AscendC::DataCopyExtParams offset_copy{1, sizeof(uint32_t), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint32_t> offset_pad{false, 0, 0, 0};
    AscendC::DataCopyPad(offset_local, offsets_gm_[block_idx_], offset_copy,
                         offset_pad);
    offset_queue_.EnQue(offset_local);
    offset_local = offset_queue_.DeQue<uint32_t>();
    uint32_t write_pos = offset_local.GetValue(0);
    offset_queue_.FreeTensor(offset_local);

    uint32_t base = begin_;
    while (base < end_) {
      const uint32_t remaining = end_ - base;
      const uint32_t length =
          remaining < kCooGetRowChunkElements ? remaining
                                               : kCooGetRowChunkElements;
      const uint32_t bytes =
          length * static_cast<uint32_t>(sizeof(IdType));
      AscendC::DataCopyExtParams copy{1, bytes, 0, 0, 0};
      AscendC::DataCopyPadExtParams<IdType> pad{false, 0, 0, 0};

      AscendC::LocalTensor<IdType> rows_local =
          rows_queue_.AllocTensor<IdType>();
      AscendC::DataCopyPad(rows_local, rows_gm_[base], copy, pad);
      rows_queue_.EnQue(rows_local);
      rows_local = rows_queue_.DeQue<IdType>();

      AscendC::LocalTensor<IdType> cols_local =
          cols_queue_.AllocTensor<IdType>();
      AscendC::DataCopyPad(cols_local, cols_gm_[base], copy, pad);
      cols_queue_.EnQue(cols_local);
      cols_local = cols_queue_.DeQue<IdType>();

      AscendC::LocalTensor<IdType> data_local;
      if (has_data_) {
        data_local = data_queue_.AllocTensor<IdType>();
        AscendC::DataCopyPad(data_local, data_gm_[base], copy, pad);
        data_queue_.EnQue(data_local);
        data_local = data_queue_.DeQue<IdType>();
      }

      AscendC::LocalTensor<IdType> picked_data =
          out_data_queue_.AllocTensor<IdType>();
      AscendC::LocalTensor<IdType> picked_cols =
          out_cols_queue_.AllocTensor<IdType>();
      uint32_t picked = 0;
      for (uint32_t i = 0; i < length; ++i) {
        if (rows_local.GetValue(i) == target_row_) {
          picked_cols.SetValue(picked, cols_local.GetValue(i));
          picked_data.SetValue(
              picked, has_data_ ? data_local.GetValue(i)
                                : static_cast<IdType>(base + i));
          ++picked;
        }
      }

      out_data_queue_.EnQue(picked_data);
      out_cols_queue_.EnQue(picked_cols);
      picked_data = out_data_queue_.DeQue<IdType>();
      picked_cols = out_cols_queue_.DeQue<IdType>();
      if (picked != 0) {
        const uint32_t output_bytes =
            picked * static_cast<uint32_t>(sizeof(IdType));
        AscendC::DataCopyExtParams output_copy{1, output_bytes, 0, 0, 0};
        AscendC::DataCopyPad(out_data_gm_[write_pos], picked_data,
                             output_copy);
        AscendC::DataCopyPad(out_cols_gm_[write_pos], picked_cols,
                             output_copy);
        write_pos += picked;
      }

      rows_queue_.FreeTensor(rows_local);
      cols_queue_.FreeTensor(cols_local);
      if (has_data_) {
        data_queue_.FreeTensor(data_local);
      }
      out_data_queue_.FreeTensor(picked_data);
      out_cols_queue_.FreeTensor(picked_cols);
      base += length;
    }
  }

 private:
  AscendC::TPipe pipe_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> rows_queue_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> cols_queue_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> data_queue_;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_data_queue_;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_cols_queue_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> offset_queue_;
  AscendC::GlobalTensor<IdType> rows_gm_;
  AscendC::GlobalTensor<IdType> cols_gm_;
  AscendC::GlobalTensor<IdType> data_gm_;
  AscendC::GlobalTensor<uint32_t> offsets_gm_;
  AscendC::GlobalTensor<IdType> out_data_gm_;
  AscendC::GlobalTensor<IdType> out_cols_gm_;
  uint32_t nnz_ = 0;
  IdType target_row_ = 0;
  bool has_data_ = false;
  uint32_t block_idx_ = 0;
  uint32_t block_num_ = 1;
  uint32_t begin_ = 0;
  uint32_t end_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void coo_get_row_count_i32(
    GM_ADDR rows, GM_ADDR count_slabs, GM_ADDR tiling) {
  CountKernel<int32_t> op;
  op.Init(rows, count_slabs, tiling);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_get_row_scatter_i32(
    GM_ADDR rows, GM_ADDR cols, GM_ADDR data, GM_ADDR offsets,
    GM_ADDR out_data, GM_ADDR out_cols, GM_ADDR tiling) {
  ScatterKernel<int32_t> op;
  op.Init(rows, cols, data, offsets, out_data, out_cols, tiling);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_get_row_count_i64(
    GM_ADDR rows, GM_ADDR count_slabs, GM_ADDR tiling) {
  CountKernel<int64_t> op;
  op.Init(rows, count_slabs, tiling);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_get_row_scatter_i64(
    GM_ADDR rows, GM_ADDR cols, GM_ADDR data, GM_ADDR offsets,
    GM_ADDR out_data, GM_ADDR out_cols, GM_ADDR tiling) {
  ScatterKernel<int64_t> op;
  op.Init(rows, cols, data, offsets, out_data, out_cols, tiling);
  op.Process();
}
