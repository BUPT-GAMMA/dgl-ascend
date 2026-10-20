/**
 * Copyright (c) 2024 by Contributors
 * @file coo_get_row_data_and_indices_tiling.h
 * @brief Host/device ABI for the COO row-selection kernels.
 */
#ifndef DGL_ARRAY_ASCEND_COO_GET_ROW_DATA_AND_INDICES_TILING_H_
#define DGL_ARRAY_ASCEND_COO_GET_ROW_DATA_AND_INDICES_TILING_H_

#include <cstdint>

constexpr uint32_t kCooGetRowChunkElements = 2048;
// Count output is block-major.  Each block owns one 32-byte slab; word 0 is
// its match count and words 1..7 are padding.  The host exclusive-scans the
// word-0 values into a uint32_t offsets array of length block_dim + 1.
constexpr uint32_t kCooGetRowCountStrideWords = 8;
// The verified standalone launch policy is min(kCooGetRowMaxBlocks, nnz).
// Empty COO input must be handled by the host without launching a kernel.
constexpr uint32_t kCooGetRowMaxBlocks = 8;

template <typename IdType>
struct CooGetRowTiling {
  uint32_t nnz;       // row/col/data element count
  IdType target_row;  // compared without narrowing
  uint32_t has_data;  // 1 = explicit data; 0 = original COO position as EID
  uint32_t reserved;
};

static_assert(sizeof(CooGetRowTiling<int32_t>) == 16,
              "unexpected int32 tiling layout");
static_assert(alignof(CooGetRowTiling<int32_t>) == 4,
              "unexpected int32 tiling alignment");
static_assert(sizeof(CooGetRowTiling<int64_t>) == 24,
              "unexpected int64 tiling layout");
static_assert(alignof(CooGetRowTiling<int64_t>) == 8,
              "unexpected int64 tiling alignment");

#endif  // DGL_ARRAY_ASCEND_COO_GET_ROW_DATA_AND_INDICES_TILING_H_
