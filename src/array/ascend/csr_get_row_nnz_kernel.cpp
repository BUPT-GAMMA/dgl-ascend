#include "kernel_operator.h"

using namespace AscendC;

template <typename IdxT>
class KernelCsrGetRowNNZ {
 public:
  __aicore__ inline KernelCsrGetRowNNZ() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR rows, GM_ADDR out, uint32_t n,
      uint32_t orig_rows) {
    // Typed __gm__ pointers throughout: per-word GlobalTensor
    // GetValue/SetValue is NOT consistently visible across blocks — the
    // multi-core probe showed chunking was correct while the degrees
    // came back corrupted, pinning the data path (rows/indptr/out) as
    // the culprit. The plain uniform kernel solved the same class of
    // problem the same way for its tables.
    indptr_p_ = (const __gm__ IdxT *)indptr;
    rows_p_ = (const __gm__ IdxT *)rows;
    out_p_ = (__gm__ IdxT *)out;
    n_ = n;
    orig_rows_ = orig_rows;
  }

  __aicore__ inline void Process() {
    const uint32_t block_id = GetBlockIdx();
    const uint32_t block_num = GetBlockNum();
    const uint32_t chunk = (n_ + block_num - 1) / block_num;
    const uint32_t start = block_id * chunk;
    const uint32_t end = (start + chunk > n_) ? n_ : start + chunk;
    for (uint32_t i = start; i < end; i++) {
      const IdxT row = rows_p_[i];
      if (static_cast<uint32_t>(row) >= orig_rows_) {
        out_p_[i] = 0;
        continue;
      }
      const IdxT start_i = indptr_p_[row];
      const IdxT end_i = indptr_p_[row + 1];
      out_p_[i] = end_i - start_i;
    }
  }

 private:
  const __gm__ IdxT *indptr_p_ = nullptr;
  const __gm__ IdxT *rows_p_ = nullptr;
  __gm__ IdxT *out_p_ = nullptr;
  uint32_t n_;
  uint32_t orig_rows_;
};

extern "C" __global__ __aicore__ void csr_get_row_nnz_int32(
    GM_ADDR indptr, GM_ADDR rows, GM_ADDR out, GM_ADDR tiling_ptr) {
  // Struct-pointer tiling access: per-word GetValue on a shared
  // GlobalTensor is not consistently visible across blocks.
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
