#include "kernel_operator.h"

using namespace AscendC;

template <typename IdxT>
class KernelCsrGetRowNNZ {
 public:
  __aicore__ inline KernelCsrGetRowNNZ() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR rows, GM_ADDR out, uint32_t n,
      uint32_t orig_rows) {
    indptr_gm.SetGlobalBuffer((__gm__ IdxT *)indptr, orig_rows + 1);
    rows_gm.SetGlobalBuffer((__gm__ IdxT *)rows, n);
    out_gm.SetGlobalBuffer((__gm__ IdxT *)out, n);
    n_ = n;
    orig_rows_ = orig_rows;
  }

  __aicore__ inline void Process() {
    uint32_t block_id = GetBlockIdx();
    uint32_t block_num = GetBlockNum();
    // TODO(tmp-probe): remove — dump per-block chunking to see whether
    // blocks disagree on n_/block_num or overlap ranges.
    AscendC::printf("[nnz-probe] blk=%u/%u n=%u chunk=%u range=[%u,%u)\n",
                    block_id, block_num, n_,
                    (n_ + block_num - 1) / block_num,
                    block_id * ((n_ + block_num - 1) / block_num),
                    (block_id * ((n_ + block_num - 1) / block_num) +
                             ((n_ + block_num - 1) / block_num) > n_)
                        ? n_
                        : 0);
    uint32_t chunk = (n_ + block_num - 1) / block_num;
    uint32_t start = block_id * chunk;
    uint32_t end = (start + chunk > n_) ? n_ : start + chunk;
    for (uint32_t i = start; i < end; i++) {
      IdxT row = rows_gm.GetValue(i);
      if (static_cast<uint32_t>(row) >= orig_rows_) {
        out_gm.SetValue(i, 0);
        continue;
      }
      IdxT start_i = indptr_gm.GetValue(static_cast<uint32_t>(row));
      IdxT end_i = indptr_gm.GetValue(static_cast<uint32_t>(row) + 1);
      out_gm.SetValue(i, end_i - start_i);
    }
  }

 private:
  GlobalTensor<IdxT> indptr_gm;
  GlobalTensor<IdxT> rows_gm;
  GlobalTensor<IdxT> out_gm;
  uint32_t n_;
  uint32_t orig_rows_;
};

extern "C" __global__ __aicore__ void csr_get_row_nnz_int32(
    GM_ADDR indptr, GM_ADDR rows, GM_ADDR out, GM_ADDR tiling_ptr) {
  // Struct-pointer tiling access (the plain uniform kernel's pattern):
  // per-word GetValue on a shared GlobalTensor is NOT consistently
  // visible across blocks — multi-block launches read inconsistent
  // n/orig_rows and corrupt the chunking (the earlier multi-core
  // attempt failed exactly here and was reverted).
  const __gm__ uint32_t *tiling = (const __gm__ uint32_t *)tiling_ptr;
  const uint32_t n = tiling[0];
  const uint32_t orig_rows = tiling[1];
  KernelCsrGetRowNNZ<int32_t> op;
  op.Init(indptr, rows, out, n, orig_rows);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_get_row_nnz_int64(
    GM_ADDR indptr, GM_ADDR rows, GM_ADDR out, GM_ADDR tiling_ptr) {
  const __gm__ uint32_t *tiling = (const __gm__ uint32_t *)tiling_ptr;
  const uint32_t n = tiling[0];
  const uint32_t orig_rows = tiling[1];
  KernelCsrGetRowNNZ<int64_t> op;
  op.Init(indptr, rows, out, n, orig_rows);
  op.Process();
}
