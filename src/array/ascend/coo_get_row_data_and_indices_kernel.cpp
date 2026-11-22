#include "kernel_operator.h"

#include "coo_get_row_data_and_indices_tiling.h"

namespace {

using namespace dgl::aten::impl::coo_get_row_v2;

__aicore__ inline void GetBlockRange(
    uint32_t nnz, uint32_t block_idx, uint32_t block_num,
    uint32_t* begin, uint32_t* end) {
  const uint32_t per_block = nnz / block_num;
  const uint32_t remainder = nnz % block_num;
  const uint32_t extra_before = block_idx < remainder ? block_idx : remainder;
  *begin = block_idx * per_block + extra_before;
  *end = *begin + per_block + (block_idx < remainder ? 1U : 0U);
}

template <AscendC::HardEvent event>
__aicore__ inline void SyncPipes() {
  const AscendC::TEventID event_id = GetTPipePtr()->FetchEventID(event);
  AscendC::SetFlag<event>(event_id);
  AscendC::WaitFlag<event>(event_id);
}

__aicore__ inline uint64_t ValidBitsMask(uint32_t valid_bits) {
  if (valid_bits >= 64U) {
    return UINT64_MAX;
  }
  if (valid_bits == 0U) {
    return 0U;
  }
  return (static_cast<uint64_t>(1) << valid_bits) - 1U;
}

__aicore__ inline uint64_t ExpandBitsToPairs(uint32_t value) {
  uint64_t expanded = value;
  expanded = (expanded | (expanded << 16)) & 0x0000FFFF0000FFFFULL;
  expanded = (expanded | (expanded << 8)) & 0x00FF00FF00FF00FFULL;
  expanded = (expanded | (expanded << 4)) & 0x0F0F0F0F0F0F0F0FULL;
  expanded = (expanded | (expanded << 2)) & 0x3333333333333333ULL;
  expanded = (expanded | (expanded << 1)) & 0x5555555555555555ULL;
  return expanded | (expanded << 1);
}

__aicore__ inline void ZeroExtendPickedEidsToI64(
    const AscendC::LocalTensor<int64_t>& dst,
    const AscendC::LocalTensor<uint32_t>& src, uint32_t count) {
  if (count == 0U) {
    return;
  }

  // DAV-2201 supports vector int32_t -> int64_t Cast, but not
  // uint32_t -> int64_t Cast.  Widen the gathered bit patterns first, then
  // clear every high 32-bit word using a vector masked Duplicate.  This is an
  // exact uint32_t zero extension, including [0x80000000, 0xffffffff].
  AscendC::Cast(
      dst, src.ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_NONE,
      count);
  AscendC::PipeBarrier<PIPE_V>();

  uint64_t high_word_mask[1] = {0xAAAAAAAAAAAAAAAAULL};
  const uint8_t repeat_times = static_cast<uint8_t>(
      (2U * count + kI32ElementsPerRepeat - 1U) /
      kI32ElementsPerRepeat);
  AscendC::Duplicate<uint32_t>(
      dst.ReinterpretCast<uint32_t>(), 0U, high_word_mask, repeat_times, 1, 8);
}

template <typename IdType>
class CountKernel {
 public:
  __aicore__ inline void Init(
      GM_ADDR rows, GM_ADDR count_slabs, GM_ADDR mask_workspace,
      GM_ADDR tiling) {
    const __gm__ GetRowTiling<IdType>* params =
        reinterpret_cast<const __gm__ GetRowTiling<IdType>*>(tiling);
    nnz_ = params->nnz;
    target_row_ = params->target_row;
    mask_chunks_per_block_ = params->mask_chunks_per_block;
    block_idx_ = AscendC::GetBlockIdx();
    block_num_ = AscendC::GetBlockNum();
    GetBlockRange(nnz_, block_idx_, block_num_, &begin_, &end_);

    rows_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(rows), nnz_);
    counts_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint32_t*>(count_slabs));
    mask_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint32_t*>(mask_workspace));
    pipe_.InitBuffer(rows_queue_, 1, kChunkElements * sizeof(IdType));
    pipe_.InitBuffer(count_buffer_, kCountStrideWords * sizeof(uint32_t));
    pipe_.InitBuffer(compact_mask_buffer_, kChunkElements / 8);
    pipe_.InitBuffer(gather_mask_buffer_, kMaskWorkspaceBytesPerChunk);
    if constexpr (sizeof(IdType) == sizeof(int64_t)) {
      pipe_.InitBuffer(low_words_buffer_, kChunkElements * sizeof(uint32_t));
      pipe_.InitBuffer(high_words_buffer_, kChunkElements * sizeof(uint32_t));
      pipe_.InitBuffer(high_mask_buffer_, kChunkElements / 8);
    }
  }

  __aicore__ inline void Process() {
    uint32_t count = 0;
    uint32_t base = begin_;
    uint32_t chunk_index = 0;
    while (base < end_) {
      const uint32_t remaining = end_ - base;
      const uint32_t length =
          remaining < kChunkElements ? remaining : kChunkElements;
      const uint32_t bytes =
          length * static_cast<uint32_t>(sizeof(IdType));

      AscendC::LocalTensor<IdType> local =
          rows_queue_.AllocTensor<IdType>();
      AscendC::DataCopyExtParams copy{1, bytes, 0, 0, 0};
      AscendC::DataCopyPadExtParams<IdType> pad{false, 0, 0, 0};
      AscendC::DataCopyPad(local, rows_gm_[base], copy, pad);
      rows_queue_.EnQue(local);
      local = rows_queue_.DeQue<IdType>();

      AscendC::LocalTensor<uint8_t> compact_mask =
          compact_mask_buffer_.Get<uint8_t>();
      AscendC::LocalTensor<uint32_t> gather_mask =
          gather_mask_buffer_.Get<uint32_t>();
      AscendC::Duplicate<uint32_t>(
          gather_mask, 0U, kMaskWorkspaceWordsPerChunk);

      if constexpr (sizeof(IdType) == sizeof(int32_t)) {
        AscendC::Compares(
            compact_mask, local, target_row_, AscendC::CMPMODE::EQ,
            kChunkElements);
      } else {
        AscendC::LocalTensor<uint32_t> low_words =
            low_words_buffer_.Get<uint32_t>();
        AscendC::LocalTensor<uint32_t> high_words =
            high_words_buffer_.Get<uint32_t>();
        AscendC::LocalTensor<uint8_t> high_mask =
            high_mask_buffer_.Get<uint8_t>();
        AscendC::LocalTensor<uint32_t> row_words =
            local.template ReinterpretCast<uint32_t>();

        uint64_t ignored_count = 0;
        AscendC::GatherMask(
            low_words, row_words, static_cast<uint8_t>(1), false, 0,
            {1, 2 * kMaskWordsPerChunk, 8, 0}, ignored_count);
        AscendC::GatherMask(
            high_words, row_words, static_cast<uint8_t>(2), false, 0,
            {1, 2 * kMaskWordsPerChunk, 8, 0}, ignored_count);
        AscendC::PipeBarrier<PIPE_V>();

        const uint64_t target_bits = static_cast<uint64_t>(target_row_);
        const int32_t target_low = static_cast<int32_t>(
            static_cast<uint32_t>(target_bits));
        const int32_t target_high = static_cast<int32_t>(
            static_cast<uint32_t>(target_bits >> 32));
        AscendC::Compares(
            compact_mask, low_words.template ReinterpretCast<int32_t>(),
            target_low, AscendC::CMPMODE::EQ, kChunkElements);
        AscendC::Compares(
            high_mask, high_words.template ReinterpretCast<int32_t>(),
            target_high, AscendC::CMPMODE::EQ, kChunkElements);
      }

      SyncPipes<AscendC::HardEvent::V_S>();
      AscendC::LocalTensor<uint64_t> compact_u64 =
          compact_mask.template ReinterpretCast<uint64_t>();
      AscendC::LocalTensor<uint64_t> gather_u64 =
          gather_mask.template ReinterpretCast<uint64_t>();
      AscendC::LocalTensor<uint64_t> high_u64;
      if constexpr (sizeof(IdType) == sizeof(int64_t)) {
        high_u64 = high_mask_buffer_.Get<uint64_t>();
      }

      for (uint32_t word_index = 0;
           word_index < kMaskWordsPerChunk; ++word_index) {
        uint64_t word = compact_u64.GetValue(word_index);
        if constexpr (sizeof(IdType) == sizeof(int64_t)) {
          word &= high_u64.GetValue(word_index);
        }
        const uint32_t word_begin = word_index * 64U;
        const uint32_t valid_bits =
            word_begin >= length
                ? 0U
                : ((length - word_begin) < 64U ? length - word_begin : 64U);
        word &= ValidBitsMask(valid_bits);
        count += static_cast<uint32_t>(AscendC::GetBitCount<1>(word));
        gather_u64.SetValue(word_index * kMaskSlabU64, word);
      }

      SyncPipes<AscendC::HardEvent::S_MTE3>();
      const uint32_t mask_word_offset =
          (block_idx_ * mask_chunks_per_block_ + chunk_index) *
          kMaskWorkspaceWordsPerChunk;
      AscendC::DataCopyExtParams mask_copy{
          1, kMaskWorkspaceBytesPerChunk, 0, 0, 0};
      AscendC::DataCopyPad(
          mask_gm_[mask_word_offset], gather_mask, mask_copy);
      SyncPipes<AscendC::HardEvent::MTE3_V>();

      rows_queue_.FreeTensor(local);
      base += length;
      ++chunk_index;
    }

    AscendC::LocalTensor<uint32_t> count_local =
        count_buffer_.Get<uint32_t>();
    for (uint32_t i = 0; i < kCountStrideWords; ++i) {
      count_local.SetValue(i, 0);
    }
    count_local.SetValue(0, count);
    SyncPipes<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyExtParams copy{
        1, kCountStrideWords * sizeof(uint32_t), 0, 0, 0};
    AscendC::DataCopyPad(
        counts_gm_[block_idx_ * kCountStrideWords], count_local, copy);
  }

 private:
  AscendC::TPipe pipe_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> rows_queue_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> count_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> compact_mask_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> gather_mask_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> low_words_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> high_words_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> high_mask_buffer_;
  AscendC::GlobalTensor<IdType> rows_gm_;
  AscendC::GlobalTensor<uint32_t> counts_gm_;
  AscendC::GlobalTensor<uint32_t> mask_gm_;
  uint32_t nnz_ = 0;
  IdType target_row_ = 0;
  uint32_t block_idx_ = 0;
  uint32_t block_num_ = 1;
  uint32_t begin_ = 0;
  uint32_t end_ = 0;
  uint32_t mask_chunks_per_block_ = 0;
};

template <typename IdType>
class ScatterKernel {
 public:
  __aicore__ inline void Init(
      GM_ADDR cols, GM_ADDR data, GM_ADDR offsets, GM_ADDR mask_workspace,
      GM_ADDR out_data, GM_ADDR out_cols, GM_ADDR tiling) {
    const __gm__ GetRowTiling<IdType>* params =
        reinterpret_cast<const __gm__ GetRowTiling<IdType>*>(tiling);
    nnz_ = params->nnz;
    has_data_ = params->has_data != 0;
    mask_chunks_per_block_ = params->mask_chunks_per_block;
    block_idx_ = AscendC::GetBlockIdx();
    block_num_ = AscendC::GetBlockNum();
    GetBlockRange(nnz_, block_idx_, block_num_, &begin_, &end_);

    cols_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(cols), nnz_);
    if (has_data_) {
      data_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ IdType*>(data), nnz_);
    }
    offsets_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint32_t*>(offsets));
    mask_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint32_t*>(mask_workspace));
    out_data_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ IdType*>(out_data));
    out_cols_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ IdType*>(out_cols));

    pipe_.InitBuffer(cols_queue_, 1, kChunkElements * sizeof(IdType));
    pipe_.InitBuffer(data_queue_, 1, kChunkElements * sizeof(IdType));
    pipe_.InitBuffer(out_data_queue_, 1, kChunkElements * sizeof(IdType));
    pipe_.InitBuffer(out_cols_queue_, 1, kChunkElements * sizeof(IdType));
    pipe_.InitBuffer(offset_queue_, 1, 32);
    pipe_.InitBuffer(mask_queue_, 1, kMaskWorkspaceBytesPerChunk);
    pipe_.InitBuffer(eid_source_buffer_, kChunkElements * sizeof(int32_t));
    if constexpr (sizeof(IdType) == sizeof(int64_t)) {
      pipe_.InitBuffer(
          expanded_mask_buffer_,
          kExpandedMaskWordsPerChunk * sizeof(uint32_t));
      pipe_.InitBuffer(
          eid_picked_buffer_, kChunkElements * sizeof(uint32_t));
    }
  }

  __aicore__ inline void Process() {
    AscendC::LocalTensor<uint32_t> offset_local =
        offset_queue_.AllocTensor<uint32_t>();
    AscendC::DataCopyExtParams offset_copy{1, sizeof(uint32_t), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint32_t> offset_pad{false, 0, 0, 0};
    AscendC::DataCopyPad(
        offset_local, offsets_gm_[block_idx_], offset_copy, offset_pad);
    offset_queue_.EnQue(offset_local);
    offset_local = offset_queue_.DeQue<uint32_t>();
    uint32_t write_pos = offset_local.GetValue(0);
    offset_queue_.FreeTensor(offset_local);

    uint32_t base = begin_;
    uint32_t chunk_index = 0;
    while (base < end_) {
      const uint32_t remaining = end_ - base;
      const uint32_t length =
          remaining < kChunkElements ? remaining : kChunkElements;
      const uint32_t bytes =
          length * static_cast<uint32_t>(sizeof(IdType));
      AscendC::DataCopyExtParams copy{1, bytes, 0, 0, 0};
      AscendC::DataCopyPadExtParams<IdType> pad{false, 0, 0, 0};

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

      AscendC::LocalTensor<uint32_t> logical_mask =
          mask_queue_.AllocTensor<uint32_t>();
      const uint32_t mask_word_offset =
          (block_idx_ * mask_chunks_per_block_ + chunk_index) *
          kMaskWorkspaceWordsPerChunk;
      AscendC::DataCopyExtParams mask_copy{
          1, kMaskWorkspaceBytesPerChunk, 0, 0, 0};
      AscendC::DataCopyPadExtParams<uint32_t> mask_pad{false, 0, 0, 0};
      AscendC::DataCopyPad(
          logical_mask, mask_gm_[mask_word_offset], mask_copy, mask_pad);
      mask_queue_.EnQue(logical_mask);
      logical_mask = mask_queue_.DeQue<uint32_t>();

      AscendC::LocalTensor<IdType> picked_data =
          out_data_queue_.AllocTensor<IdType>();
      AscendC::LocalTensor<IdType> picked_cols =
          out_cols_queue_.AllocTensor<IdType>();
      uint32_t picked = 0;
      if constexpr (sizeof(IdType) == sizeof(int32_t)) {
        uint64_t picked_cols_count = 0;
        AscendC::GatherMask(
            picked_cols, cols_local, logical_mask, true,
            kI32ElementsPerRepeat,
            {1, kMaskWordsPerChunk, 8, 1}, picked_cols_count);

        uint64_t picked_data_count = 0;
        if (has_data_) {
          AscendC::GatherMask(
              picked_data, data_local, logical_mask, true,
              kI32ElementsPerRepeat,
              {1, kMaskWordsPerChunk, 8, 1}, picked_data_count);
        } else {
          AscendC::LocalTensor<int32_t> eid_source =
              eid_source_buffer_.Get<int32_t>();
          AscendC::CreateVecIndex(
              eid_source, static_cast<int32_t>(base), kChunkElements);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::GatherMask(
              picked_data, eid_source, logical_mask, true,
              kI32ElementsPerRepeat,
              {1, kMaskWordsPerChunk, 8, 1}, picked_data_count);
        }
        picked = static_cast<uint32_t>(picked_cols_count);
      } else {
        AscendC::LocalTensor<uint32_t> expanded_mask =
            expanded_mask_buffer_.Get<uint32_t>();
        AscendC::Duplicate<uint32_t>(
            expanded_mask, 0U, kExpandedMaskWordsPerChunk);
        SyncPipes<AscendC::HardEvent::MTE2_S>();
        SyncPipes<AscendC::HardEvent::V_S>();

        AscendC::LocalTensor<uint64_t> logical_u64 =
            logical_mask.template ReinterpretCast<uint64_t>();
        AscendC::LocalTensor<uint64_t> expanded_u64 =
            expanded_mask.template ReinterpretCast<uint64_t>();
        for (uint32_t word_index = 0;
             word_index < kMaskWordsPerChunk; ++word_index) {
          const uint64_t logical =
              logical_u64.GetValue(word_index * kMaskSlabU64);
          expanded_u64.SetValue(
              (2 * word_index) * kMaskSlabU64,
              ExpandBitsToPairs(static_cast<uint32_t>(logical)));
          expanded_u64.SetValue(
              (2 * word_index + 1) * kMaskSlabU64,
              ExpandBitsToPairs(static_cast<uint32_t>(logical >> 32)));
        }
        SyncPipes<AscendC::HardEvent::S_V>();

        uint64_t picked_col_words = 0;
        AscendC::GatherMask(
            picked_cols.template ReinterpretCast<uint32_t>(),
            cols_local.template ReinterpretCast<uint32_t>(),
            expanded_mask, true, kI32ElementsPerRepeat,
            {1, 2 * kMaskWordsPerChunk, 8, 1}, picked_col_words);
        picked = static_cast<uint32_t>(picked_col_words / 2U);

        if (has_data_) {
          uint64_t picked_data_words = 0;
          AscendC::GatherMask(
              picked_data.template ReinterpretCast<uint32_t>(),
              data_local.template ReinterpretCast<uint32_t>(),
              expanded_mask, true, kI32ElementsPerRepeat,
              {1, 2 * kMaskWordsPerChunk, 8, 1}, picked_data_words);
        } else {
          AscendC::LocalTensor<int32_t> eid_source =
              eid_source_buffer_.Get<int32_t>();
          AscendC::LocalTensor<uint32_t> eid_picked =
              eid_picked_buffer_.Get<uint32_t>();
          AscendC::CreateVecIndex(
              eid_source, static_cast<int32_t>(base), kChunkElements);
          AscendC::PipeBarrier<PIPE_V>();
          uint64_t picked_eids = 0;
          AscendC::GatherMask(
              eid_picked, eid_source.template ReinterpretCast<uint32_t>(),
              logical_mask, true, kI32ElementsPerRepeat,
              {1, kMaskWordsPerChunk, 8, 1}, picked_eids);
          AscendC::PipeBarrier<PIPE_V>();
          ZeroExtendPickedEidsToI64(picked_data, eid_picked, picked);
        }
      }

      out_data_queue_.EnQue(picked_data);
      out_cols_queue_.EnQue(picked_cols);
      picked_data = out_data_queue_.DeQue<IdType>();
      picked_cols = out_cols_queue_.DeQue<IdType>();
      SyncPipes<AscendC::HardEvent::S_MTE3>();
      if (picked != 0) {
        const uint32_t output_bytes =
            picked * static_cast<uint32_t>(sizeof(IdType));
        AscendC::DataCopyExtParams output_copy{
            1, output_bytes, 0, 0, 0};
        AscendC::DataCopyPad(
            out_data_gm_[write_pos], picked_data, output_copy);
        AscendC::DataCopyPad(
            out_cols_gm_[write_pos], picked_cols, output_copy);
        write_pos += picked;
      }

      cols_queue_.FreeTensor(cols_local);
      if (has_data_) {
        data_queue_.FreeTensor(data_local);
      }
      out_data_queue_.FreeTensor(picked_data);
      out_cols_queue_.FreeTensor(picked_cols);
      mask_queue_.FreeTensor(logical_mask);
      base += length;
      ++chunk_index;
    }
  }

 private:
  AscendC::TPipe pipe_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> cols_queue_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> data_queue_;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_data_queue_;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_cols_queue_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> offset_queue_;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> mask_queue_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> expanded_mask_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> eid_source_buffer_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> eid_picked_buffer_;
  AscendC::GlobalTensor<IdType> cols_gm_;
  AscendC::GlobalTensor<IdType> data_gm_;
  AscendC::GlobalTensor<uint32_t> offsets_gm_;
  AscendC::GlobalTensor<uint32_t> mask_gm_;
  AscendC::GlobalTensor<IdType> out_data_gm_;
  AscendC::GlobalTensor<IdType> out_cols_gm_;
  uint32_t nnz_ = 0;
  bool has_data_ = false;
  uint32_t block_idx_ = 0;
  uint32_t block_num_ = 1;
  uint32_t begin_ = 0;
  uint32_t end_ = 0;
  uint32_t mask_chunks_per_block_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void coo_get_row_count_i32(
    GM_ADDR rows, GM_ADDR count_slabs, GM_ADDR mask_workspace,
    GM_ADDR tiling) {
  CountKernel<int32_t> op;
  op.Init(rows, count_slabs, mask_workspace, tiling);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_get_row_scatter_i32(
    GM_ADDR cols, GM_ADDR data, GM_ADDR offsets, GM_ADDR mask_workspace,
    GM_ADDR out_data, GM_ADDR out_cols, GM_ADDR tiling) {
  ScatterKernel<int32_t> op;
  op.Init(cols, data, offsets, mask_workspace, out_data, out_cols, tiling);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_get_row_count_i64(
    GM_ADDR rows, GM_ADDR count_slabs, GM_ADDR mask_workspace,
    GM_ADDR tiling) {
  CountKernel<int64_t> op;
  op.Init(rows, count_slabs, mask_workspace, tiling);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_get_row_scatter_i64(
    GM_ADDR cols, GM_ADDR data, GM_ADDR offsets, GM_ADDR mask_workspace,
    GM_ADDR out_data, GM_ADDR out_cols, GM_ADDR tiling) {
  ScatterKernel<int64_t> op;
  op.Init(cols, data, offsets, mask_workspace, out_data, out_cols, tiling);
  op.Process();
}
