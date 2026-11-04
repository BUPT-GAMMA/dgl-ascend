#pragma once

#include <cstdint>

namespace dgl {
namespace aten {
namespace impl {
namespace coo_get_row_v2 {

constexpr uint32_t kChunkElements = 2048;
constexpr uint32_t kCountStrideWords = 8;  // one 32-byte slab per block
constexpr uint32_t kMaxBlocks = 8;
constexpr uint32_t kVectorBytes = 256;
constexpr uint32_t kDataBlockBytes = 32;
constexpr uint32_t kI32ElementsPerRepeat =
    kVectorBytes / sizeof(int32_t);
constexpr uint32_t kMaskWordsPerChunk = kChunkElements / 64;
constexpr uint32_t kMaskSlabWords =
    kDataBlockBytes / sizeof(uint32_t);
constexpr uint32_t kMaskSlabU64 =
    kDataBlockBytes / sizeof(uint64_t);
constexpr uint32_t kMaskWorkspaceWordsPerChunk =
    kMaskWordsPerChunk * kMaskSlabWords;
constexpr uint32_t kMaskWorkspaceBytesPerChunk =
    kMaskWorkspaceWordsPerChunk * sizeof(uint32_t);
constexpr uint32_t kExpandedMaskWordsPerChunk =
    2 * kMaskWorkspaceWordsPerChunk;

// Shared verbatim by the host compiler and AscendC device compiler.  Keep the
// static assertions next to the definition so either side fails immediately if
// a compiler or field edit changes the ABI.
template <typename IdType>
struct GetRowTiling {
  uint32_t nnz;
  IdType target_row;
  uint32_t has_data;
  uint32_t mask_chunks_per_block;
};

static_assert(
    sizeof(GetRowTiling<int32_t>) == 16,
    "GetRowTiling<int32_t> layout changed");
static_assert(
    alignof(GetRowTiling<int32_t>) == 4,
    "GetRowTiling<int32_t> alignment changed");
static_assert(
    sizeof(GetRowTiling<int64_t>) == 24,
    "GetRowTiling<int64_t> layout changed");
static_assert(
    alignof(GetRowTiling<int64_t>) == 8,
    "GetRowTiling<int64_t> alignment changed");

}  // namespace coo_get_row_v2
}  // namespace impl
}  // namespace aten
}  // namespace dgl
