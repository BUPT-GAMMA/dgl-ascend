/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_sampling_uniform_kernel.cpp
 * @brief AscendC kernel for uniform CSR row-wise sampling on Ascend NPU.
 *
 * Single-core (blockDim=1) scalar-GM kernel — verified correct (16/16 tests).
 *
 * Multi-core attempts (2026-08-12):
 * - VECOUT SetValue + DataCopyPad output: proven reliable (mc_vecout_test
 *   5/5 PASS, 2560/2560 correct). One large AllocTensor per block + one
 *   DataCopyPad works for output.
 * - VECIN DataCopyPad + EnQue/DeQue for input: produces ~25% corrupted data
 *   when scalar GetValue reads from the DeQue'd tensor. Root cause unknown —
 *   EnQue/DeQue should sync MTE2→Vector but scalar reads get garbage.
 * - Workaround: scalar GM GetValue for input (correct but 85x slower than
 *   single-core due to per-element GM reads across 40 cores).
 * - Single-core scalar-GM remains the shipped path (correct + 2.4ms/call).
 *
 * Next step to unlock multi-core: investigate why VECIN DataCopyPad +
 * EnQue/DeQue produces corrupted data on scalar GetValue. Possible: need
 * PipeBarrier after DeQue, or VECIN TQue depth/buffer config issue.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {

__aicore__ inline uint32_t xorshift32(uint32_t& x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

__aicore__ inline uint32_t rand_below(uint32_t& state, uint32_t n) {
  if (n == 0) return 0;
  uint32_t r = xorshift32(state);
  return static_cast<uint32_t>((static_cast<uint64_t>(r) * n) >> 32);
}

}  // namespace

template <typename IdT>
class KernelCsrRowWiseSamplingUniform {
 public:
  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
      GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
      uint32_t num_rows, uint32_t num_samples, uint32_t replace,
      uint32_t has_data, uint32_t seed, uint32_t select_all,
      uint32_t num_total_rows) {
    indptr_gm.SetGlobalBuffer((__gm__ IdT*)indptr);
    indices_gm.SetGlobalBuffer((__gm__ IdT*)indices);
    if (has_data) data_gm.SetGlobalBuffer((__gm__ IdT*)data);
    rows_gm.SetGlobalBuffer((__gm__ IdT*)rows, num_rows);
    out_ptr_gm.SetGlobalBuffer((__gm__ IdT*)out_ptr, num_rows + 1);
    out_rows_gm.SetGlobalBuffer((__gm__ IdT*)out_rows);
    out_cols_gm.SetGlobalBuffer((__gm__ IdT*)out_cols);
    out_idxs_gm.SetGlobalBuffer((__gm__ IdT*)out_idxs);
    num_rows_ = num_rows;
    num_samples_ = num_samples;
    replace_ = replace;
    has_data_ = has_data;
    seed_ = seed;
    select_all_ = select_all;
    num_total_rows_ = num_total_rows;
  }

  __aicore__ inline void Process() {
    uint32_t offset = 0;
    for (uint32_t i = 0; i < num_rows_; ++i) {
      out_ptr_gm.SetValue(i, static_cast<IdT>(offset));
      IdT rid = rows_gm.GetValue(i);
      // Defensive: drop out-of-range row ids (negative or >= num_total_rows)
      // as empty rows instead of reading indptr out of bounds. Matches the
      // CPU path, where COOSliceRows silently discards invalid seed ids.
      if (rid < 0 || rid >= static_cast<IdT>(num_total_rows_)) {
        continue;
      }
      IdT off = indptr_gm.GetValue(static_cast<uint32_t>(rid));
      IdT end = indptr_gm.GetValue(static_cast<uint32_t>(rid) + 1);
      uint32_t deg = static_cast<uint32_t>(end - off);
      uint32_t num_picks = select_all_ ? deg :
          (replace_ ? (deg == 0 ? 0 : num_samples_) :
           (deg < num_samples_ ? deg : num_samples_));
      if (num_picks > 0) {
        uint32_t state = seed_ ^ (i * 2654435761u + 0x9e3779b9u);
        if (state == 0) state = 0x12345678u;
        if (select_all_ || (!replace_ && num_picks == deg)) {
          for (uint32_t j = 0; j < num_picks; ++j)
            WritePick(offset + j, rid, off + static_cast<IdT>(j));
        } else if (replace_) {
          for (uint32_t j = 0; j < num_picks; ++j) {
            uint32_t idx = rand_below(state, deg);
            WritePick(offset + j, rid, off + static_cast<IdT>(idx));
          }
        } else {
          for (uint32_t j = 0; j < num_picks; ++j)
            out_idxs_gm.SetValue(offset + j, static_cast<IdT>(j));
          for (uint32_t i2 = num_picks; i2 < deg; ++i2) {
            uint32_t j = rand_below(state, i2 + 1);
            if (j < num_picks)
              out_idxs_gm.SetValue(offset + j, static_cast<IdT>(i2));
          }
          for (uint32_t j = 0; j < num_picks; ++j) {
            IdT local = out_idxs_gm.GetValue(offset + j);
            IdT picked = off + local;
            out_rows_gm.SetValue(offset + j, rid);
            out_cols_gm.SetValue(offset + j, indices_gm.GetValue(static_cast<uint32_t>(picked)));
            out_idxs_gm.SetValue(offset + j, has_data_ ? data_gm.GetValue(static_cast<uint32_t>(picked)) : picked);
          }
        }
      }
      offset += num_picks;
    }
    out_ptr_gm.SetValue(num_rows_, static_cast<IdT>(offset));
  }

 private:
  __aicore__ inline void WritePick(uint32_t pos, IdT rid, IdT picked) {
    out_rows_gm.SetValue(pos, rid);
    out_cols_gm.SetValue(pos, indices_gm.GetValue(static_cast<uint32_t>(picked)));
    out_idxs_gm.SetValue(pos, has_data_ ? data_gm.GetValue(static_cast<uint32_t>(picked)) : picked);
  }
  GlobalTensor<IdT> indptr_gm, indices_gm, data_gm, rows_gm, out_ptr_gm, out_rows_gm, out_cols_gm, out_idxs_gm;
  uint32_t num_rows_ = 0, num_samples_ = 0, replace_ = 0, has_data_ = 0, seed_ = 0, select_all_ = 0;
  uint32_t num_total_rows_ = 0;
};

extern "C" __global__ __aicore__ void csr_row_wise_sampling_uniform_int32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
    GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
    GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 7);
  KernelCsrRowWiseSamplingUniform<int32_t> op;
  op.Init(indptr, indices, data, rows, out_ptr, out_rows, out_cols, out_idxs,
          tilingGm.GetValue(0), tilingGm.GetValue(1), tilingGm.GetValue(2),
          tilingGm.GetValue(3), tilingGm.GetValue(4), tilingGm.GetValue(5),
          tilingGm.GetValue(6));
  op.Process();
}

extern "C" __global__ __aicore__ void csr_row_wise_sampling_uniform_int64(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
    GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
    GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 7);
  KernelCsrRowWiseSamplingUniform<int64_t> op;
  op.Init(indptr, indices, data, rows, out_ptr, out_rows, out_cols, out_idxs,
          tilingGm.GetValue(0), tilingGm.GetValue(1), tilingGm.GetValue(2),
          tilingGm.GetValue(3), tilingGm.GetValue(4), tilingGm.GetValue(5),
          tilingGm.GetValue(6));
  op.Process();
}
