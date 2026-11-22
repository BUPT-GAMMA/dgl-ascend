#ifndef DGL_ARRAY_ASCEND_COO_GET_ROW_DATA_AND_INDICES_TILING_H_
#define DGL_ARRAY_ASCEND_COO_GET_ROW_DATA_AND_INDICES_TILING_H_

#include <cstdint>

constexpr uint32_t kCooGetRowTile = 2048;
constexpr uint32_t kCooGetRowCountWords = 8;

template <typename IdType>
struct CooGetRowTiling {
  uint32_t nnz;   
  IdType targetRow; 
  uint32_t hasData;
  uint32_t reserved;
};


#endif
