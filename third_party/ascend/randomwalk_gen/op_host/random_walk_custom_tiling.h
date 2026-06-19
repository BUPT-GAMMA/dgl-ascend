#ifndef RANDOM_WALK_CUSTOM_TILING_H
#define RANDOM_WALK_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/pad/pad_tilingdata.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RandomWalkCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, dataType);
TILING_DATA_FIELD_DEF(uint32_t, dataSize);

TILING_DATA_FIELD_DEF(uint32_t, seedsDataLength);
TILING_DATA_FIELD_DEF(uint32_t, rowPtrLength);
TILING_DATA_FIELD_DEF(uint32_t, colIndLength);
TILING_DATA_FIELD_DEF(uint32_t, metapathDataLength);
TILING_DATA_FIELD_DEF(uint32_t, randomNumberDataLength);
TILING_DATA_FIELD_DEF(uint32_t, outputLength);

TILING_DATA_FIELD_DEF(uint32_t, seedsDataLengthAligned);
TILING_DATA_FIELD_DEF(uint32_t, rowPtrLengthAligned);
TILING_DATA_FIELD_DEF(uint32_t, colIndLengthAligned);
TILING_DATA_FIELD_DEF(uint32_t, metapathDataLengthAligned);
TILING_DATA_FIELD_DEF(uint32_t, randomNumberDataLengthAligned);
TILING_DATA_FIELD_DEF(uint32_t, outputLengthAligned);

TILING_DATA_FIELD_DEF(uint32_t, ubLeftSize);
TILING_DATA_FIELD_DEF(uint32_t, baseBlocksPerCore);
TILING_DATA_FIELD_DEF_STRUCT(UnPadTiling, UnPadTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RandomWalkCustom, RandomWalkCustomTilingData)
}
#endif
