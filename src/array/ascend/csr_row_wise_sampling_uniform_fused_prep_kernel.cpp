/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_sampling_uniform_fused_prep_kernel.cpp
 * @brief Single-core preparation kernel for uniform CSR fused sampling.
 *
 * Replaces the host-side pick/partition chain (degree D2H → CPU pick
 * computation → balanced partition → block-offset prefix sums → packed
 * table upload) with one device pass over the degree array that
 * CSRGetRowNNZ already produced. Single core on purpose: the workload is
 * a linear 100k-element scan (~1ms), far below the multi-block
 * coordination cost, and a single writer keeps row_split/out_starts
 * trivially consistent.
 *
 * Outputs (all device-resident, consumed directly by the sampling
 * kernel through typed __gm__ pointers):
 *   picks[num_rows]      per-row pick counts (uint32)
 *   row_split[block_dim + 1]  nnz-balanced row-range boundaries
 *   out_starts[block_dim + 1] per-block output offsets; the final entry
 *                             doubles as max_output for the host's
 *                             output allocation (one scalar D2H)
 *
 * The balancing mirrors BuildBalancedPartitionsFused exactly: prefix
 * targets at total*part/block_dim, first index whose prefix >= target,
 * ties resolved to the same boundary (lower_bound semantics).
 */

#include "csr_row_wise_sampling_uniform_fused_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t kPrepWindow = 8192;  // degrees staged per GM read

}  // namespace

template <typename DegT>
class KernelFusedPrep {
 public:
  __aicore__ inline KernelFusedPrep() {}

  __aicore__ inline void Init(
      GM_ADDR deg, GM_ADDR picks, GM_ADDR row_split, GM_ADDR out_starts,
      GM_ADDR tiling_ptr, TPipe* pipe) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const __gm__ uint32_t* tiling = (const __gm__ uint32_t*)tiling_ptr;
    num_rows_ = tiling[0];
    fanout_ = tiling[1];
    replace_ = tiling[2];
    select_all_ = tiling[3];
    block_dim_ = tiling[4];

    // Degrees arrive with the graph's idtype (int32/int64) — read them
    // through the matching type or every other word is garbage.
    deg_gm_.SetGlobalBuffer((__gm__ DegT*)deg, num_rows_);
    picks_gm_.SetGlobalBuffer((__gm__ uint32_t*)picks, num_rows_);
    row_split_gm_.SetGlobalBuffer((__gm__ uint32_t*)row_split, block_dim_ + 1);
    out_starts_gm_.SetGlobalBuffer(
        (__gm__ uint32_t*)out_starts, block_dim_ + 1);

    pipe->InitBuffer(win_buf_, 2, kPrepWindow * sizeof(DegT));
    pipe->InitBuffer(pick_buf_, kPrepWindow * sizeof(uint32_t));
    pipe->InitBuffer(split_buf_, (kMaxBlocks + 1) * sizeof(uint32_t));
    pipe->InitBuffer(starts_buf_, (kMaxBlocks + 1) * sizeof(uint32_t));
  }

  __aicore__ inline void Process() {
    // Pass over the degrees in windows: compute picks, stage them for a
    // batched write, and track the running prefix. row_split/out_starts
    // are small (< 42 entries) and written once at the end through the
    // MTE path (multi-word GM scalar stores are banned family-wide).
    LocalTensor<uint32_t> pick_win = pick_buf_.Get<uint32_t>();
    LocalTensor<uint32_t> split_local = split_buf_.Get<uint32_t>();

    uint64_t prefix = 0;     // running pick total
    uint32_t next_part = 1;  // next boundary index to resolve
    uint32_t split_idx = 0;  // rows consumed by resolved boundaries
    split_local.SetValue(0, 0);

    uint32_t row = 0;
    while (row < num_rows_) {
      const uint32_t avail = num_rows_ - row;
      const uint32_t count = avail < kPrepWindow ? avail : kPrepWindow;
      const uint32_t copy_bytes = count * sizeof(uint32_t);
      DataCopyExtParams cp{1, copy_bytes, 0, 0, 0};
      DataCopyPadExtParams<DegT> pad{false, 0, 0, 0};
      LocalTensor<DegT> deg_win = win_buf_.AllocTensor<DegT>();
      DataCopyPad(deg_win, deg_gm_[row], cp, pad);
      win_buf_.EnQue(deg_win);
      deg_win = win_buf_.DeQue<DegT>();

      for (uint32_t j = 0; j < count; ++j) {
        const uint32_t d = static_cast<uint32_t>(deg_win.GetValue(j));
        const uint32_t p = select_all_ ? d
                           : replace_  ? (d == 0 ? 0u : fanout_)
                                       : (d < fanout_ ? d : fanout_);
        pick_win.SetValue(j, p);
      }

      // Boundaries resolve in the second pass below (targets need the
      // final total); this loop is a pure scan.

      // Batched picks write (MTE path for cross-block consistency).
      {
        const uint32_t write_bytes = count * sizeof(uint32_t);
        DataCopyExtParams wcp{1, write_bytes, 0, 0, 0};
        DataCopyPad(picks_gm_[row], pick_win, wcp);
      }
      for (uint32_t j = 0; j < count; ++j) {
        prefix += pick_win.GetValue(j);
      }
      row += count;
      win_buf_.FreeTensor(deg_win);
    }

    // Boundary resolution needs targets = total*part/block_dim, which
    // needs the full prefix — rescan picks from GM (sequential reads,
    // single core, windowed). The MTE3 writes above must drain before
    // the scalar reads below see them (V-side read after MTE-side write
    // needs the pipeline barrier; without it the rescan reads stale GM).
    AscendC::PipeBarrier<PIPE_MTE3>();
    const uint32_t total = static_cast<uint32_t>(prefix);
    uint32_t boundary = 0;  // candidate row index
    uint64_t run2 = 0;      // second-pass prefix
    uint32_t part = 1;
    uint32_t resolved = 1;  // entries filled in split_local (incl. [0])
    split_local.SetValue(0, 0);
    while (boundary < num_rows_ && part < block_dim_) {
      const uint32_t target = static_cast<uint32_t>(
          (static_cast<uint64_t>(total) * part) / block_dim_);
      if (total == 0) {
        // Degenerate: equal row splits (matches the host's zero-weight
        // fallback).
        for (; part < block_dim_; ++part, ++resolved) {
          split_local.SetValue(
              resolved,
              static_cast<uint32_t>(
                  (static_cast<uint64_t>(part) * num_rows_) / block_dim_));
        }
        break;
      }
      if (run2 < target) {
        // advance one row at a time until the prefix reaches the target
        run2 += picks_gm_.GetValue(boundary);
        ++boundary;
      } else {
        // boundary is the first index with prefix >= target
        split_local.SetValue(resolved, boundary);
        ++resolved;
        ++part;
      }
    }
    for (; resolved <= block_dim_; ++resolved) {
      split_local.SetValue(resolved, num_rows_);
    }
    // Guard monotonicity against ties (mirrors the host path).
    for (uint32_t i = 1; i <= block_dim_; ++i) {
      const uint32_t prev = split_local.GetValue(i - 1);
      const uint32_t cur = split_local.GetValue(i);
      if (cur < prev) split_local.SetValue(i, prev);
    }

    // out_starts[b] = prefix at row_split[b]; one more scan building
    // both tables' final GM images through the MTE path.
    LocalTensor<uint32_t> starts_local = starts_buf_.Get<uint32_t>();
    uint64_t run3 = 0;
    uint32_t sb = 0;  // index into split_local
    uint32_t next_split = split_local.GetValue(0);
    for (uint32_t i = 0; i <= num_rows_; ++i) {
      while (sb < block_dim_ && i == next_split) {
        starts_local.SetValue(sb, static_cast<uint32_t>(run3));
        ++sb;
        next_split = split_local.GetValue(sb);
      }
      if (i < num_rows_) run3 += picks_gm_.GetValue(i);
    }
    starts_local.SetValue(block_dim_, static_cast<uint32_t>(run3));

    {
      const uint32_t bytes = (block_dim_ + 1) * sizeof(uint32_t);
      DataCopyExtParams cp1{1, bytes, 0, 0, 0};
      (void)cp1;
      DataCopyPad(row_split_gm_[0], split_local, cp1);
      DataCopyExtParams cp2{1, bytes, 0, 0, 0};
      DataCopyPad(out_starts_gm_[0], starts_local, cp2);
    }
  }

 private:
  static constexpr uint32_t kMaxBlocks = 64;  // row_split capacity guard

  const __gm__ uint32_t* deg_p_ = nullptr;  // scalar reads (single core)
  GlobalTensor<DegT> deg_gm_;
  GlobalTensor<uint32_t> picks_gm_, row_split_gm_, out_starts_gm_;
  uint32_t num_rows_ = 0, fanout_ = 0, replace_ = 0, select_all_ = 0,
           block_dim_ = 0;
  TQue<TPosition::VECIN, 2> win_buf_;
  TBuf<TPosition::VECCALC> pick_buf_, split_buf_, starts_buf_;
};

extern "C" __global__ __aicore__ void
csr_row_wise_sampling_uniform_fused_prep_int32(
    GM_ADDR deg, GM_ADDR picks, GM_ADDR row_split, GM_ADDR out_starts,
    GM_ADDR tiling_ptr) {
  KernelFusedPrep<int32_t> op;
  TPipe pipe;
  op.Init(deg, picks, row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void
csr_row_wise_sampling_uniform_fused_prep_int64(
    GM_ADDR deg, GM_ADDR picks, GM_ADDR row_split, GM_ADDR out_starts,
    GM_ADDR tiling_ptr) {
  KernelFusedPrep<int64_t> op;
  TPipe pipe;
  op.Init(deg, picks, row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}
