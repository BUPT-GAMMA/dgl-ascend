#include "kernel_operator.h"

#include <cstdint>

using namespace AscendC;

namespace {

constexpr uint32_t kCOOQueryTileLength = 256;
constexpr uint32_t kCOOQueryMaskBytes = kCOOQueryTileLength / 8;
constexpr uint32_t kCOOGatherFlushElements = 256;

static_assert(
    kCOOQueryMaskBytes == 32,
    "Each match tile must produce one aligned mask block");

// Keep the layout identical to coo_get_data_and_indices.cc.
struct COOGetDataAndIndicesTiling {
  uint32_t nnz;
  uint32_t padded_nnz;
  uint32_t query_count;
  uint32_t reverse_order;
  uint32_t has_data;
};

static_assert(
    sizeof(COOGetDataAndIndicesTiling) == 20,
    "Unexpected COOGetDataAndIndices tiling layout");

template <HardEvent Event>
__aicore__ inline void COOQuerySync(TPipe* pipe) {
  const int32_t event_id =
      static_cast<int32_t>(pipe->FetchEventID(Event));
  SetFlag<Event>(event_id);
  WaitFlag<Event>(event_id);
}

template <typename IdType>
class KernelCOOQueryMatch {
 public:
  __aicore__ inline void Init(
      GM_ADDR coo_rows, GM_ADDR coo_cols,
      GM_ADDR query_rows, GM_ADDR query_cols,
      GM_ADDR mask, GM_ADDR tiling_ptr, TPipe* pipe) {
    pipe_ = pipe;

    const __gm__ COOGetDataAndIndicesTiling* tiling =
        reinterpret_cast<
            const __gm__ COOGetDataAndIndicesTiling*>(tiling_ptr);

    padded_nnz_ = tiling->padded_nnz;
    query_count_ = tiling->query_count;
    mask_stride_ = padded_nnz_ / 8;

    coo_rows_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ IdType*>(coo_rows),
        padded_nnz_);
    coo_cols_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ IdType*>(coo_cols),
        padded_nnz_);

    query_rows_ =
        reinterpret_cast<const __gm__ IdType*>(query_rows);
    query_cols_ =
        reinterpret_cast<const __gm__ IdType*>(query_cols);

    mask_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t*>(mask),
        static_cast<uint64_t>(query_count_) * mask_stride_);

    pipe_->InitBuffer(
        in_rows_, 1, kCOOQueryTileLength * sizeof(IdType));
    pipe_->InitBuffer(
        in_cols_, 1, kCOOQueryTileLength * sizeof(IdType));
    pipe_->InitBuffer(out_mask_, 1, kCOOQueryMaskBytes);
    pipe_->InitBuffer(row_mask_, kCOOQueryMaskBytes);
    pipe_->InitBuffer(col_mask_, kCOOQueryMaskBytes);
  }

  __aicore__ inline void Process() {
    const uint32_t block_count = GetBlockNum();

    for (uint64_t q = GetBlockIdx();
         q < query_count_;
         q += block_count) {
      const IdType query_row = query_rows_[q];
      const IdType query_col = query_cols_[q];
      const uint64_t output_base = q * mask_stride_;

      for (uint32_t start = 0;
           start < padded_nnz_;
           start += kCOOQueryTileLength) {
        LocalTensor<IdType> rows =
            in_rows_.AllocTensor<IdType>();
        LocalTensor<IdType> cols =
            in_cols_.AllocTensor<IdType>();

        DataCopy(
            rows, coo_rows_gm_[start],
            kCOOQueryTileLength);
        DataCopy(
            cols, coo_cols_gm_[start],
            kCOOQueryTileLength);

        in_rows_.EnQue(rows);
        in_cols_.EnQue(cols);

        rows = in_rows_.DeQue<IdType>();
        cols = in_cols_.DeQue<IdType>();

        LocalTensor<uint8_t> mask =
            out_mask_.AllocTensor<uint8_t>();

        ComputeMask(
            rows, cols, query_row, query_col, mask);

        out_mask_.EnQue(mask);
        in_rows_.FreeTensor(rows);
        in_cols_.FreeTensor(cols);

        mask = out_mask_.DeQue<uint8_t>();

        DataCopy(
            mask_gm_[output_base + start / 8],
            mask, kCOOQueryMaskBytes);

        out_mask_.FreeTensor(mask);
      }
    }
  }

 private:
  __aicore__ inline void ComputeMask(
      const LocalTensor<IdType>& rows,
      const LocalTensor<IdType>& cols,
      IdType query_row, IdType query_col,
      LocalTensor<uint8_t>& mask);

  TPipe* pipe_ = nullptr;

  TQue<TPosition::VECIN, 1> in_rows_;
  TQue<TPosition::VECIN, 1> in_cols_;
  TQue<TPosition::VECOUT, 1> out_mask_;

  TBuf<TPosition::VECCALC> row_mask_;
  TBuf<TPosition::VECCALC> col_mask_;

  GlobalTensor<IdType> coo_rows_gm_;
  GlobalTensor<IdType> coo_cols_gm_;
  GlobalTensor<uint8_t> mask_gm_;

  const __gm__ IdType* query_rows_ = nullptr;
  const __gm__ IdType* query_cols_ = nullptr;

  uint32_t padded_nnz_ = 0;
  uint32_t query_count_ = 0;
  uint32_t mask_stride_ = 0;
};

template <>
__aicore__ inline void
KernelCOOQueryMatch<int32_t>::ComputeMask(
    const LocalTensor<int32_t>& rows,
    const LocalTensor<int32_t>& cols,
    int32_t query_row, int32_t query_col,
    LocalTensor<uint8_t>& mask) {
  LocalTensor<uint8_t> row_mask =
      row_mask_.Get<uint8_t>();
  LocalTensor<uint8_t> col_mask =
      col_mask_.Get<uint8_t>();

  CompareScalar(
      row_mask, rows, query_row,
      CMPMODE::EQ, kCOOQueryTileLength);
  CompareScalar(
      col_mask, cols, query_col,
      CMPMODE::EQ, kCOOQueryTileLength);

  PipeBarrier<PIPE_V>();

  And(
      mask.ReinterpretCast<uint16_t>(),
      row_mask.ReinterpretCast<uint16_t>(),
      col_mask.ReinterpretCast<uint16_t>(),
      static_cast<int32_t>(
          kCOOQueryMaskBytes / sizeof(uint16_t)));

  PipeBarrier<PIPE_V>();
}

template <>
__aicore__ inline void
KernelCOOQueryMatch<int64_t>::ComputeMask(
    const LocalTensor<int64_t>& rows,
    const LocalTensor<int64_t>& cols,
    int64_t query_row, int64_t query_col,
    LocalTensor<uint8_t>& mask) {
  // Queues synchronize vector-side consumers. The scalar path explicitly
  // synchronizes its own reads and writes with the DMA pipelines.
  COOQuerySync<HardEvent::MTE2_S>(pipe_);
  COOQuerySync<HardEvent::MTE3_S>(pipe_);

  for (uint32_t byte = 0;
       byte < kCOOQueryMaskBytes;
       ++byte) {
    uint8_t bits = 0;

    for (uint32_t bit = 0; bit < 8; ++bit) {
      const uint32_t index = byte * 8 + bit;

      if (rows.GetValue(index) == query_row
          && cols.GetValue(index) == query_col) {
        bits = static_cast<uint8_t>(
            bits | (1U << bit));
      }
    }

    mask.SetValue(byte, bits);
  }

  COOQuerySync<HardEvent::S_MTE2>(pipe_);
  COOQuerySync<HardEvent::S_MTE3>(pipe_);
}

template <typename IdType>
class KernelCOOQueryGather {
 public:
  __aicore__ inline void Init(
      GM_ADDR query_rows, GM_ADDR query_cols,
      GM_ADDR coo_data, GM_ADDR mask, GM_ADDR offsets,
      GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_data,
      GM_ADDR tiling_ptr, TPipe* pipe) {
    pipe_ = pipe;

    const __gm__ COOGetDataAndIndicesTiling* tiling =
        reinterpret_cast<
            const __gm__ COOGetDataAndIndicesTiling*>(tiling_ptr);

    nnz_ = tiling->nnz;
    padded_nnz_ = tiling->padded_nnz;
    query_count_ = tiling->query_count;
    reverse_order_ = tiling->reverse_order != 0;
    has_data_ = tiling->has_data != 0;

    mask_stride_ = padded_nnz_ / 8;

    query_rows_ =
        reinterpret_cast<const __gm__ IdType*>(query_rows);
    query_cols_ =
        reinterpret_cast<const __gm__ IdType*>(query_cols);
    coo_data_ =
        reinterpret_cast<const __gm__ IdType*>(coo_data);
    offsets_ =
        reinterpret_cast<const __gm__ uint64_t*>(offsets);

    mask_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t*>(mask),
        static_cast<uint64_t>(query_count_) * mask_stride_);

    const uint64_t total = offsets_[query_count_];

    out_rows_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ IdType*>(out_rows), total);
    out_cols_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ IdType*>(out_cols), total);
    out_data_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ IdType*>(out_data), total);

    pipe_->InitBuffer(mask_buf_, kCOOQueryMaskBytes);
    pipe_->InitBuffer(
        out_rows_buf_,
        kCOOGatherFlushElements * sizeof(IdType));
    pipe_->InitBuffer(
        out_cols_buf_,
        kCOOGatherFlushElements * sizeof(IdType));
    pipe_->InitBuffer(
        out_data_buf_,
        kCOOGatherFlushElements * sizeof(IdType));
  }

  __aicore__ inline void Process() {
    // Host launches one block. This guard also prevents duplicate writes
    // if a caller accidentally supplies a larger logical grid.
    if (GetBlockIdx() != 0) return;

    LocalTensor<uint8_t> mask =
        mask_buf_.Get<uint8_t>();
    LocalTensor<IdType> out_rows =
        out_rows_buf_.Get<IdType>();
    LocalTensor<IdType> out_cols =
        out_cols_buf_.Get<IdType>();
    LocalTensor<IdType> out_data =
        out_data_buf_.Get<IdType>();

    const uint32_t tile_count =
        padded_nnz_ / kCOOQueryTileLength;

    for (uint64_t q = 0; q < query_count_; ++q) {
      uint64_t output_index = offsets_[q];
      const uint64_t query_end = offsets_[q + 1];

      if (output_index == query_end) continue;

      const IdType query_row = query_rows_[q];
      const IdType query_col = query_cols_[q];
      const uint64_t mask_base = q * mask_stride_;
      uint32_t staged = 0;

      for (uint32_t ordinal = 0;
           ordinal < tile_count;
           ++ordinal) {
        const uint32_t tile_index =
            reverse_order_
                ? tile_count - 1 - ordinal
                : ordinal;

        const uint32_t tile_start =
            tile_index * kCOOQueryTileLength;

        DataCopy(
            mask,
            mask_gm_[mask_base + tile_start / 8],
            kCOOQueryMaskBytes);

        COOQuerySync<HardEvent::MTE2_S>(pipe_);

        for (uint32_t lane = 0;
             lane < kCOOQueryTileLength;
             ++lane) {
          const uint32_t local_index =
              reverse_order_
                  ? kCOOQueryTileLength - 1 - lane
                  : lane;

          const uint32_t coo_index =
              tile_start + local_index;

          // Padding may contain matching values. Never emit padded entries.
          if (coo_index >= nnz_) continue;

          const uint8_t bits =
              mask.GetValue(local_index / 8);

          if (((bits >> (local_index % 8)) & 1U) == 0) {
            continue;
          }

          out_rows.SetValue(staged, query_row);
          out_cols.SetValue(staged, query_col);

          const IdType value =
              has_data_
                  ? coo_data_[coo_index]
                  : static_cast<IdType>(coo_index);

          out_data.SetValue(staged, value);
          ++staged;

          if (staged == kCOOGatherFlushElements) {
            FlushOut(
                out_rows, out_cols, out_data,
                output_index, staged);
            output_index += staged;
            staged = 0;
          }
        }

        // Complete scalar mask reads before the next MTE2 copy reuses UB.
        COOQuerySync<HardEvent::S_MTE2>(pipe_);
      }

      if (staged != 0) {
        FlushOut(
            out_rows, out_cols, out_data,
            output_index, staged);
      }
    }
  }

 private:
  __aicore__ inline void FlushOut(
      LocalTensor<IdType>& rows,
      LocalTensor<IdType>& cols,
      LocalTensor<IdType>& data,
      uint64_t output_index, uint32_t count) {
    // Scalar SetValue must complete before MTE3 reads these buffers.
    COOQuerySync<HardEvent::S_MTE3>(pipe_);

    const uint32_t copy_bytes =
        count * static_cast<uint32_t>(sizeof(IdType));

    DataCopyExtParams copy_params{
        1, copy_bytes, 0, 0, 0};

    DataCopyPad(
        out_rows_gm_[output_index],
        rows, copy_params);
    DataCopyPad(
        out_cols_gm_[output_index],
        cols, copy_params);
    DataCopyPad(
        out_data_gm_[output_index],
        data, copy_params);

    // Complete all three copies before scalar stores reuse the buffers.
    COOQuerySync<HardEvent::MTE3_S>(pipe_);
  }

  TPipe* pipe_ = nullptr;

  TBuf<TPosition::VECCALC> mask_buf_;
  TBuf<TPosition::VECCALC> out_rows_buf_;
  TBuf<TPosition::VECCALC> out_cols_buf_;
  TBuf<TPosition::VECCALC> out_data_buf_;

  GlobalTensor<uint8_t> mask_gm_;
  GlobalTensor<IdType> out_rows_gm_;
  GlobalTensor<IdType> out_cols_gm_;
  GlobalTensor<IdType> out_data_gm_;

  const __gm__ IdType* query_rows_ = nullptr;
  const __gm__ IdType* query_cols_ = nullptr;
  const __gm__ IdType* coo_data_ = nullptr;
  const __gm__ uint64_t* offsets_ = nullptr;

  uint32_t nnz_ = 0;
  uint32_t padded_nnz_ = 0;
  uint32_t query_count_ = 0;
  uint32_t mask_stride_ = 0;

  bool reverse_order_ = false;
  bool has_data_ = false;
};

}  // namespace

extern "C" __global__ __aicore__ void
coo_get_data_and_indices_match_int32(
    GM_ADDR coo_rows, GM_ADDR coo_cols,
    GM_ADDR query_rows, GM_ADDR query_cols,
    GM_ADDR mask, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  TPipe pipe;
  KernelCOOQueryMatch<int32_t> op;
  op.Init(
      coo_rows, coo_cols, query_rows, query_cols,
      mask, tiling, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void
coo_get_data_and_indices_match_int64(
    GM_ADDR coo_rows, GM_ADDR coo_cols,
    GM_ADDR query_rows, GM_ADDR query_cols,
    GM_ADDR mask, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  TPipe pipe;
  KernelCOOQueryMatch<int64_t> op;
  op.Init(
      coo_rows, coo_cols, query_rows, query_cols,
      mask, tiling, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void
coo_get_data_and_indices_gather_int32(
    GM_ADDR query_rows, GM_ADDR query_cols,
    GM_ADDR coo_data, GM_ADDR mask, GM_ADDR offsets,
    GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_data,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  TPipe pipe;
  KernelCOOQueryGather<int32_t> op;
  op.Init(
      query_rows, query_cols, coo_data, mask, offsets,
      out_rows, out_cols, out_data, tiling, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void
coo_get_data_and_indices_gather_int64(
    GM_ADDR query_rows, GM_ADDR query_cols,
    GM_ADDR coo_data, GM_ADDR mask, GM_ADDR offsets,
    GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_data,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  TPipe pipe;
  KernelCOOQueryGather<int64_t> op;
  op.Init(
      query_rows, query_cols, coo_data, mask, offsets,
      out_rows, out_cols, out_data, tiling, &pipe);
  op.Process();
}