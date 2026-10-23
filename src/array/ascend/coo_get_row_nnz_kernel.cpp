
/**
 * COOGetRowNNZ Ascend C Kernel
 *
 * Count the number of COO entries belonging to each query row.
 *
 * Supports:
 *   int32_t
 *   int64_t
 *
 * Device kernel arguments:
 *   coo_rows : COO row indices in Ascend GM
 *   queries  : query row indices in Ascend GM
 *   output   : contiguous result array in Ascend GM
 *   tiling   : uint64_t[2] = {nnz, num_queries}
 *
 * The host must validate query row IDs against coo.num_rows.
 */

#include "kernel_operator.h"

#include <cstdint>

using namespace AscendC;

namespace {

constexpr uint32_t kTileElems = 256;
constexpr uint32_t kMaskBytes = 64;
constexpr uint32_t kCacheLineBytes = 64;
constexpr uint32_t kOutputTileElems = 256;

template <typename T>
class KernelCOOGetRowNNZ {
 public:
  __aicore__ inline KernelCOOGetRowNNZ() {}

  __aicore__ inline void Init(
      GM_ADDR coo_rows,
      GM_ADDR queries,
      GM_ADDR output,
      GM_ADDR tiling,
      TPipe* pipe) {
    // Read parameters from Ascend Global Memory.
    const __gm__ uint64_t* params =
        reinterpret_cast<const __gm__ uint64_t*>(tiling);

    nnz_ = params[0];
    num_queries_ = params[1];

    rows_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(coo_rows), nnz_);

    output_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(output), num_queries_);

    queries_ptr_ =
        reinterpret_cast<const __gm__ T*>(queries);

    // Input: 256 elements.
    pipe->InitBuffer(
        input_queue_, 1,
        kTileElems * sizeof(T));

    // Comparison masks.
    pipe->InitBuffer(mask_low_buf_, kMaskBytes);
    pipe->InitBuffer(mask_high_buf_, kMaskBytes);

    // Contiguous output staging.
    pipe->InitBuffer(
        output_buf_, kOutputTileElems * sizeof(T));
  }

  __aicore__ inline void Process() {
    if (num_queries_ == 0) {
      return;
    }

    const uint64_t block_id = GetBlockIdx();
    const uint64_t block_num = GetBlockNum();

    // Assign contiguous query ranges to each block.
    // Each range begins on a 64-byte boundary.
    const uint64_t align_elems =
        kCacheLineBytes / sizeof(T);

    const uint64_t groups =
        (num_queries_ + align_elems - 1) /
        align_elems;

    const uint64_t groups_per_block =
        (groups + block_num - 1) / block_num;

    const uint64_t queries_per_block =
        groups_per_block * align_elems;

    const uint64_t begin =
        block_id * queries_per_block;

    if (begin >= num_queries_) {
      return;
    }

    const uint64_t remaining =
        num_queries_ - begin;

    const uint64_t length =
        remaining < queries_per_block
            ? remaining
            : queries_per_block;

    const uint64_t end = begin + length;

    LocalTensor<T> output_local =
        output_buf_.Get<T>();

    uint32_t staged = 0;
    uint64_t stage_begin = begin;

    for (uint64_t q = begin; q < end; ++q) {
      const T target = queries_ptr_[q];

      const uint64_t count = CountRow(target);

      output_local.SetValue(
          staged, static_cast<T>(count));

      ++staged;

      if (staged == kOutputTileElems) {
        FlushOutput(output_local, stage_begin, staged);
        stage_begin += staged;
        staged = 0;
      }
    }

    if (staged != 0) {
      FlushOutput(output_local, stage_begin, staged);
    }
  }

 private:
  __aicore__ inline uint64_t CountRow(T target) {
    uint64_t total = 0;

    for (uint64_t offset = 0;
         offset < nnz_;
         offset += kTileElems) {

      const uint64_t remaining = nnz_ - offset;

      const uint32_t valid =
          static_cast<uint32_t>(
              remaining < kTileElems
                  ? remaining
                  : kTileElems);

      LocalTensor<T> input =
          input_queue_.AllocTensor<T>();

      // Copy only real COO elements from GM.
      DataCopyExtParams copy_params{
          1,
          static_cast<uint32_t>(valid * sizeof(T)),
          0, 0, 0
      };

      DataCopyPadExtParams<T> pad_params{
          false, 0, 0, static_cast<T>(0)
      };

      DataCopyPad(
          input,
          rows_gm_[offset],
          copy_params,
          pad_params);

      input_queue_.EnQue(input);
      input = input_queue_.DeQue<T>();

      // CompareScalar uses an aligned vector length.
      // Complete the final UB tile locally, without
      // reading past the original GM allocation.
      if (valid < kTileElems) {
        for (uint32_t i = valid;
             i < kTileElems; ++i) {
          input.SetValue(i, static_cast<T>(-1));
        }

        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
      }

      total += CountMatches(input, target, valid);

      input_queue_.FreeTensor(input);
    }

    return total;
  }

  __aicore__ inline uint32_t CountMatches(
      LocalTensor<T> input,
      T target,
      uint32_t valid) {

    LocalTensor<uint8_t> mask_low =
        mask_low_buf_.Get<uint8_t>();

    uint32_t count = 0;

    if constexpr (sizeof(T) == 4) {
      // int32_t: direct vector equality comparison.
      CompareScalar(
          mask_low,
          input,
          target,
          CMPMODE::EQ,
          kTileElems);

      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);

      for (uint32_t i = 0;
           i < kTileElems / 8; ++i) {

        const uint32_t first = i * 8;
        if (first >= valid) {
          break;
        }

        uint32_t bits = mask_low.GetValue(i);

        // Ignore padding positions in the tail.
        const uint32_t active =
            valid - first < 8
                ? valid - first
                : 8;

        bits &= (1u << active) - 1u;

        count +=
            ScalarGetCountOfValue<1>(
                static_cast<uint64_t>(bits));
      }

    } else {
      // int64_t: reinterpret each 64-bit value as
      // two consecutive 32-bit words.
      LocalTensor<int32_t> words =
          input.template ReinterpretCast<int32_t>();

      LocalTensor<uint8_t> mask_high =
          mask_high_buf_.Get<uint8_t>();

      const uint64_t raw =
          static_cast<uint64_t>(target);

      const int32_t target_low =
          static_cast<int32_t>(
              static_cast<uint32_t>(raw));

      const int32_t target_high =
          static_cast<int32_t>(
              static_cast<uint32_t>(raw >> 32));

      CompareScalar(
          mask_low,
          words,
          target_low,
          CMPMODE::EQ,
          kTileElems * 2);

      CompareScalar(
          mask_high,
          words,
          target_high,
          CMPMODE::EQ,
          kTileElems * 2);

      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);

      // 8 mask bits cover four int64_t elements.
      for (uint32_t i = 0;
           i < (kTileElems * 2) / 8; ++i) {

        const uint32_t first = i * 4;
        if (first >= valid) {
          break;
        }

        const uint32_t low_bits =
            mask_low.GetValue(i);

        const uint32_t high_bits =
            mask_high.GetValue(i);

        // Low and high 32-bit words must both match.
        uint32_t matched =
            low_bits & (high_bits >> 1) & 0x55u;

        const uint32_t active =
            valid - first < 4
                ? valid - first
                : 4;

        const uint32_t valid_mask =
            ((1u << (2 * active)) - 1u) & 0x55u;

        matched &= valid_mask;

        count +=
            ScalarGetCountOfValue<1>(
                static_cast<uint64_t>(matched));
      }
    }

    // Finish Scalar mask reads before the next
    // Vector comparison reuses the buffers.
    SetFlag<HardEvent::S_V>(EVENT_ID1);
    WaitFlag<HardEvent::S_V>(EVENT_ID1);

    return count;
  }


__aicore__ inline void FlushOutput(
    LocalTensor<T>& staging,
    uint64_t offset,
    uint32_t count) {

  DataCopyExtParams copy_params{
      1,
      static_cast<uint32_t>(count * sizeof(T)),
      0, 0, 0
  };

  // Wait until Scalar writes to UB are visible to MTE3.
  const auto s_to_mte3 =
      GetTPipePtr()->FetchEventID(HardEvent::S_MTE3);

  SetFlag<HardEvent::S_MTE3>(s_to_mte3);
  WaitFlag<HardEvent::S_MTE3>(s_to_mte3);

  DataCopyPad(
      output_gm_[offset],
      staging,
      copy_params);

  // Wait until DMA finishes reading the staging buffer
  // before Scalar reuses the same UB memory.
  const auto mte3_to_s =
      GetTPipePtr()->FetchEventID(HardEvent::MTE3_S);

  SetFlag<HardEvent::MTE3_S>(mte3_to_s);
  WaitFlag<HardEvent::MTE3_S>(mte3_to_s);
}


 private:
  uint64_t nnz_ = 0;
  uint64_t num_queries_ = 0;

  const __gm__ T* queries_ptr_ = nullptr;

  GlobalTensor<T> rows_gm_;
  GlobalTensor<T> output_gm_;

  TQue<TPosition::VECIN, 1> input_queue_;

  TBuf<TPosition::VECCALC> mask_low_buf_;
  TBuf<TPosition::VECCALC> mask_high_buf_;
  TBuf<TPosition::VECCALC> output_buf_;
};

}  // namespace

// =====================================================
// int32_t Kernel entry
// =====================================================

extern "C" __global__ __aicore__
void coo_get_row_nnz_int32(
    GM_ADDR coo_rows,
    GM_ADDR queries,
    GM_ADDR output,
    GM_ADDR tiling) {

  TPipe pipe;
  KernelCOOGetRowNNZ<int32_t> op;

  op.Init(
      coo_rows, queries, output, tiling, &pipe);

  op.Process();
}

// =====================================================
// int64_t Kernel entry
// =====================================================

extern "C" __global__ __aicore__
void coo_get_row_nnz_int64(
    GM_ADDR coo_rows,
    GM_ADDR queries,
    GM_ADDR output,
    GM_ADDR tiling) {

  TPipe pipe;
  KernelCOOGetRowNNZ<int64_t> op;

  op.Init(
      coo_rows, queries, output, tiling, &pipe);

  op.Process();
}
