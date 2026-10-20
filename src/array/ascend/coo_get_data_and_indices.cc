#ifdef DGL_USE_ASCEND

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <dgl/array.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "../array_op.h"

// Large-query duplicate ordering was validated on GCC 9 libstdc++.
// Other standard libraries retain the native CPU ordering below instead of
// preventing the entire Ascend backend from compiling.

extern "C" uint32_t
aclrtlaunch_coo_get_data_and_indices_match_int32(
    uint32_t block_dim, aclrtStream stream,
    void* coo_rows, void* coo_cols,
    void* query_rows, void* query_cols,
    void* mask, void* tiling);

extern "C" uint32_t
aclrtlaunch_coo_get_data_and_indices_match_int64(
    uint32_t block_dim, aclrtStream stream,
    void* coo_rows, void* coo_cols,
    void* query_rows, void* query_cols,
    void* mask, void* tiling);

extern "C" uint32_t
aclrtlaunch_coo_get_data_and_indices_gather_int32(
    uint32_t block_dim, aclrtStream stream,
    void* query_rows, void* query_cols,
    void* coo_data, void* mask, void* offsets,
    void* out_rows, void* out_cols, void* out_data,
    void* tiling);

extern "C" uint32_t
aclrtlaunch_coo_get_data_and_indices_gather_int64(
    uint32_t block_dim, aclrtStream stream,
    void* query_rows, void* query_cols,
    void* coo_data, void* mask, void* offsets,
    void* out_rows, void* out_cols, void* out_data,
    void* tiling);

namespace dgl {
namespace runtime {

aclrtStream getCurrentAscendStream();

}  // namespace runtime

namespace aten {
namespace impl {
namespace {

constexpr uint32_t kCOOQueryTileLength = 256;

// Keep the field order identical to the kernel-side structure.
struct COOGetDataAndIndicesTiling {
  uint32_t nnz;
  uint32_t padded_nnz;
  uint32_t query_count;
  uint32_t reverse_order;
  uint32_t has_data;
};

static_assert(
    sizeof(COOGetDataAndIndicesTiling) == 20,
    "Unexpected COOGetDataAndIndices tiling layout");

inline void CheckCOOQueryAcl(aclError status, const char* operation) {
  CHECK_EQ(status, ACL_SUCCESS)
      << operation << " failed, aclError=" << status;
}

inline size_t COOQueryBytes(uint64_t count, size_t element_bytes) {
  CHECK_LE(
      count,
      static_cast<uint64_t>(
          std::numeric_limits<size_t>::max() / element_bytes))
      << "COOGetDataAndIndices byte size overflow";
  return static_cast<size_t>(count) * element_bytes;
}

// Own only this operator's temporary allocations. The DGL stream and device
// lifecycle remain owned by DGL.
class COOQueryScratch {
 public:
  explicit COOQueryScratch(aclrtStream stream) : stream_(stream) {}

  COOQueryScratch(const COOQueryScratch&) = delete;
  COOQueryScratch& operator=(const COOQueryScratch&) = delete;

  void* Allocate(size_t bytes) {
    CHECK_GT(bytes, 0U);
    buffers_.push_back(nullptr);
    CheckCOOQueryAcl(
        aclrtMalloc(
            &buffers_.back(), bytes, ACL_MEM_MALLOC_HUGE_FIRST),
        "aclrtMalloc");
    return buffers_.back();
  }

  void MarkPending() {
    pending_ = true;
  }

  void Synchronize() {
    CheckCOOQueryAcl(
        aclrtSynchronizeStream(stream_),
        "aclrtSynchronizeStream");
    pending_ = false;
  }

  void ReleaseChecked() {
    if (pending_) Synchronize();

    aclError first_error = ACL_SUCCESS;
    for (auto it = buffers_.rbegin(); it != buffers_.rend(); ++it) {
      if (*it == nullptr) continue;
      void* pointer = *it;
      *it = nullptr;
      const aclError status = aclrtFree(pointer);
      if (status != ACL_SUCCESS && first_error == ACL_SUCCESS) {
        first_error = status;
      }
    }

    CheckCOOQueryAcl(first_error, "aclrtFree");
  }

  ~COOQueryScratch() {
    if (pending_) {
      const aclError status = aclrtSynchronizeStream(stream_);
      if (status != ACL_SUCCESS) {
        // DGL owns the device. Do not reset it, or free memory that an
        // unsuccessful asynchronous task might still be accessing.
        LOG(ERROR)
            << "COOGetDataAndIndices cleanup synchronization failed: "
            << status
            << "; temporary allocations retained to avoid unsafe frees";
        return;
      }
    }

    for (auto it = buffers_.rbegin(); it != buffers_.rend(); ++it) {
      if (*it == nullptr) continue;
      const aclError status = aclrtFree(*it);
      if (status != ACL_SUCCESS) {
        LOG(ERROR)
            << "COOGetDataAndIndices cleanup aclrtFree failed: "
            << status;
      }
    }
  }

 private:
  aclrtStream stream_;
  bool pending_ = false;
  std::vector<void*> buffers_;
};

template <typename IdType>
void CheckCOOQueryArray(
    const NDArray& array, DGLContext ctx, const char* name) {
  // COO and query index arrays are one-dimensional.
  CHECK_EQ(array->ndim, 1) << name << " must be one-dimensional";
  CHECK_GE(array->shape[0], 0) << name << " has a negative length";
  CHECK_EQ(array->dtype.code, kDGLInt)
      << name << " must contain signed integer indices";
  CHECK_EQ(array->dtype.bits, sizeof(IdType) * 8)
      << name << " has an unexpected integer width";
  CHECK_EQ(array->dtype.lanes, 1)
      << name << " must have one dtype lane";
  CHECK_EQ(array->ctx.device_type, ctx.device_type)
      << name << " is on a different device type";
  CHECK_EQ(array->ctx.device_id, ctx.device_id)
      << name << " is on a different device";
}

template <typename IdType>
void CopyExpandedCOOQueries(
    const NDArray& input, std::vector<IdType>* expanded) {
  if (input->shape[0] == 1) {
    IdType value = 0;
    CheckCOOQueryAcl(
        aclrtMemcpy(
            &value, sizeof(value), input->data, sizeof(value),
            ACL_MEMCPY_DEVICE_TO_HOST),
        "aclrtMemcpy query scalar D2H");
    std::fill(expanded->begin(), expanded->end(), value);
  } else {
    const size_t bytes =
        COOQueryBytes(expanded->size(), sizeof(IdType));
    CheckCOOQueryAcl(
        aclrtMemcpy(
            expanded->data(), bytes, input->data, bytes,
            ACL_MEMCPY_DEVICE_TO_HOST),
        "aclrtMemcpy queries D2H");
  }
}

template <typename IdType>
std::vector<NDArray> COOGetDataAndIndicesAscend(
    COOMatrix coo, NDArray rows, NDArray cols) {
  static_assert(
      std::is_same<IdType, int32_t>::value
          || std::is_same<IdType, int64_t>::value,
      "Unsupported COO index type");

  const DGLContext ctx = coo.row->ctx;
  CHECK_EQ(ctx.device_type, kDGLAscend)
      << "Expected Ascend COO input";

  CheckCOOQueryArray<IdType>(coo.row, ctx, "COO row");
  CheckCOOQueryArray<IdType>(coo.col, ctx, "COO col");
  CheckCOOQueryArray<IdType>(rows, ctx, "query rows");
  CheckCOOQueryArray<IdType>(cols, ctx, "query cols");

  const int64_t nnz = coo.row->shape[0];
  CHECK_EQ(coo.col->shape[0], nnz)
      << "COO row and col lengths must match";

  const bool has_data = COOHasData(coo);
  if (has_data) {
    CheckCOOQueryArray<IdType>(coo.data, ctx, "COO data");
    CHECK_EQ(coo.data->shape[0], nnz)
        << "COO data length must equal nnz";
  }

  const int64_t rowlen = rows->shape[0];
  const int64_t collen = cols->shape[0];
  const int64_t len = std::max(rowlen, collen);

  CHECK(
      rowlen == collen || rowlen == 1 || collen == 1)
      << "Invalid row and col id array.";

  auto empty_result = [&]() -> std::vector<NDArray> {
    return {
        NDArray::Empty({0}, coo.row->dtype, ctx),
        NDArray::Empty({0}, coo.row->dtype, ctx),
        NDArray::Empty({0}, coo.row->dtype, ctx)};
  };

  // Matches the CPU paired-loop condition, including shapes (0,1) and (1,0).
  if (rowlen == 0 || collen == 0) {
    return empty_result();
  }

#if !defined(__GLIBCXX__) || __GLIBCXX__ != 20210601 \
    || !defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE != 9
  // Equivalent-key iteration order is not portable across standard libraries.
  // Keep exact native CPU order for the hash-table branch on unvalidated ones.
  if (len >= 200) {
    const DGLContext cpu_ctx{kDGLCPU, 0};
    auto result = COOGetDataAndIndices<kDGLCPU, IdType>(
        coo.CopyTo(cpu_ctx), rows.CopyTo(cpu_ctx), cols.CopyTo(cpu_ctx));
    for (auto& array : result) array = array.CopyTo(ctx);
    return result;
  }
#endif

  CHECK_LE(
      static_cast<uint64_t>(len),
      static_cast<uint64_t>(
          std::numeric_limits<uint32_t>::max()))
      << "Query count exceeds the current kernel tiling limit";

  CHECK_LE(
      static_cast<uint64_t>(nnz),
      static_cast<uint64_t>(
          std::numeric_limits<uint32_t>::max()))
      << "COO nnz exceeds the current kernel tiling limit";

  if (!has_data && nnz != 0) {
    CHECK_LE(
        static_cast<uint64_t>(nnz - 1),
        static_cast<uint64_t>(
            std::numeric_limits<IdType>::max()))
        << "Implicit COO position exceeds the output index type";
  }

  const uint64_t query_count = static_cast<uint64_t>(len);
  const uint64_t padded_nnz =
      ((static_cast<uint64_t>(nnz) + kCOOQueryTileLength - 1)
       / kCOOQueryTileLength)
      * kCOOQueryTileLength;

  CHECK_LE(
      padded_nnz,
      static_cast<uint64_t>(
          std::numeric_limits<uint32_t>::max()))
      << "Padded COO length exceeds the current kernel tiling limit";

  const size_t query_bytes =
      COOQueryBytes(query_count, sizeof(IdType));

  std::vector<IdType> query_rows(
      static_cast<size_t>(query_count));
  std::vector<IdType> query_cols(
      static_cast<size_t>(query_count));

  CheckCOOQueryAcl(
      aclrtSetDevice(ctx.device_id), "aclrtSetDevice");

  const aclrtStream stream =
      dgl::runtime::getCurrentAscendStream();

  // Complete preceding work on the DGL stream before synchronous copies.
  CheckCOOQueryAcl(
      aclrtSynchronizeStream(stream),
      "aclrtSynchronizeStream before input copies");

  CopyExpandedCOOQueries<IdType>(rows, &query_rows);
  CopyExpandedCOOQueries<IdType>(cols, &query_cols);

  for (uint64_t q = 0; q < query_count; ++q) {
    CHECK_GE(query_rows[q], 0)
        << "Invalid row index: " << query_rows[q];
    CHECK_LT(query_rows[q], coo.num_rows)
        << "Invalid row index: " << query_rows[q];
    CHECK_GE(query_cols[q], 0)
        << "Invalid col index: " << query_cols[q];
    CHECK_LT(query_cols[q], coo.num_cols)
        << "Invalid col index: " << query_cols[q];
  }

  // Nonempty queries must still be bounds-checked for an empty COO.
  if (nnz == 0) {
    return empty_result();
  }

  const uint64_t mask_stride = padded_nnz / 8;
  // Both operands are bounded by uint32_t above; this product fits uint64_t.
  const uint64_t mask_count = query_count * mask_stride;

  const size_t input_bytes =
      COOQueryBytes(padded_nnz, sizeof(IdType));
  const size_t valid_input_bytes =
      COOQueryBytes(static_cast<uint64_t>(nnz), sizeof(IdType));
  const size_t mask_bytes = COOQueryBytes(mask_count, 1);
  const size_t offset_bytes =
      COOQueryBytes(query_count + 1, sizeof(uint64_t));

  std::vector<uint8_t> host_mask(mask_bytes);
  std::vector<uint64_t> offsets(
      static_cast<size_t>(query_count + 1), 0);

  COOGetDataAndIndicesTiling tiling{
      static_cast<uint32_t>(nnz),
      static_cast<uint32_t>(padded_nnz),
      static_cast<uint32_t>(query_count),
      len >= 200 ? 1U : 0U,
      has_data ? 1U : 0U};

  COOQueryScratch scratch(stream);

  void* padded_rows_dev = scratch.Allocate(input_bytes);
  void* padded_cols_dev = scratch.Allocate(input_bytes);
  void* query_rows_dev = scratch.Allocate(query_bytes);
  void* query_cols_dev = scratch.Allocate(query_bytes);
  void* mask_dev = scratch.Allocate(mask_bytes);
  void* offsets_dev = scratch.Allocate(offset_bytes);
  void* tiling_dev = scratch.Allocate(sizeof(tiling));

  CheckCOOQueryAcl(
      aclrtMemset(
          padded_rows_dev, input_bytes, 0, input_bytes),
      "aclrtMemset padded COO rows");
  CheckCOOQueryAcl(
      aclrtMemset(
          padded_cols_dev, input_bytes, 0, input_bytes),
      "aclrtMemset padded COO cols");

  CheckCOOQueryAcl(
      aclrtMemcpy(
          padded_rows_dev, input_bytes, coo.row->data,
          valid_input_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE),
      "aclrtMemcpy COO rows D2D");
  CheckCOOQueryAcl(
      aclrtMemcpy(
          padded_cols_dev, input_bytes, coo.col->data,
          valid_input_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE),
      "aclrtMemcpy COO cols D2D");

  CheckCOOQueryAcl(
      aclrtMemcpy(
          query_rows_dev, query_bytes, query_rows.data(),
          query_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
      "aclrtMemcpy expanded query rows H2D");
  CheckCOOQueryAcl(
      aclrtMemcpy(
          query_cols_dev, query_bytes, query_cols.data(),
          query_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
      "aclrtMemcpy expanded query cols H2D");
  CheckCOOQueryAcl(
      aclrtMemcpy(
          tiling_dev, sizeof(tiling), &tiling,
          sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE),
      "aclrtMemcpy tiling H2D");

  uint32_t block_dim = 1;
  int64_t core_count = 0;
  const aclError core_status = aclrtGetDeviceInfo(
      ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &core_count);
  if (core_status == ACL_SUCCESS
      && core_count > 0 && core_count <= 4096) {
    block_dim = static_cast<uint32_t>(
        std::min<uint64_t>(
            query_count, static_cast<uint64_t>(core_count)));
  }

  scratch.MarkPending();

  aclError match_status;
  if (std::is_same<IdType, int32_t>::value) {
    match_status =
        aclrtlaunch_coo_get_data_and_indices_match_int32(
            block_dim, stream,
            padded_rows_dev, padded_cols_dev,
            query_rows_dev, query_cols_dev,
            mask_dev, tiling_dev);
  } else {
    match_status =
        aclrtlaunch_coo_get_data_and_indices_match_int64(
            block_dim, stream,
            padded_rows_dev, padded_cols_dev,
            query_rows_dev, query_cols_dev,
            mask_dev, tiling_dev);
  }

  CheckCOOQueryAcl(
      match_status, "COOGetDataAndIndices match launch");
  scratch.Synchronize();

  CheckCOOQueryAcl(
      aclrtMemcpy(
          host_mask.data(), mask_bytes,
          mask_dev, mask_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
      "aclrtMemcpy match mask D2H");

  const uint64_t valid_mask_bytes =
      (static_cast<uint64_t>(nnz) + 7) / 8;
  const uint32_t tail_bits =
      static_cast<uint32_t>(nnz % 8);

  for (uint64_t q = 0; q < query_count; ++q) {
    uint64_t count = 0;
    const uint64_t base = q * mask_stride;

    for (uint64_t byte = 0; byte < valid_mask_bytes; ++byte) {
      uint8_t bits = host_mask[base + byte];
      if (tail_bits != 0 && byte + 1 == valid_mask_bytes) {
        bits = static_cast<uint8_t>(
            bits & ((1U << tail_bits) - 1U));
      }
      while (bits != 0) {
        count += bits & 1U;
        bits = static_cast<uint8_t>(bits >> 1);
      }
    }

    CHECK_LE(
        count,
        static_cast<uint64_t>(
            std::numeric_limits<int64_t>::max()) - offsets[q])
        << "COOGetDataAndIndices output length overflow";

    offsets[q + 1] = offsets[q] + count;
  }

  const uint64_t total = offsets.back();
  COOQueryBytes(total, sizeof(IdType));

  if (total == 0) {
    scratch.ReleaseChecked();
    return empty_result();
  }

  const int64_t output_length = static_cast<int64_t>(total);
  NDArray out_rows =
      NDArray::Empty({output_length}, coo.row->dtype, ctx);
  NDArray out_cols =
      NDArray::Empty({output_length}, coo.row->dtype, ctx);
  NDArray out_data =
      NDArray::Empty({output_length}, coo.row->dtype, ctx);

  CheckCOOQueryAcl(
      aclrtMemcpy(
          offsets_dev, offset_bytes, offsets.data(),
          offset_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
      "aclrtMemcpy output offsets H2D");

  void* coo_data_dev = has_data ? coo.data->data : nullptr;

  scratch.MarkPending();

  // Dynamic query segments may share a transfer block. Start with one
  // gather block to avoid concurrent partial writes to adjacent segments.
  aclError gather_status;
  if (std::is_same<IdType, int32_t>::value) {
    gather_status =
        aclrtlaunch_coo_get_data_and_indices_gather_int32(
            1, stream,
            query_rows_dev, query_cols_dev,
            coo_data_dev, mask_dev, offsets_dev,
            out_rows->data, out_cols->data, out_data->data,
            tiling_dev);
  } else {
    gather_status =
        aclrtlaunch_coo_get_data_and_indices_gather_int64(
            1, stream,
            query_rows_dev, query_cols_dev,
            coo_data_dev, mask_dev, offsets_dev,
            out_rows->data, out_cols->data, out_data->data,
            tiling_dev);
  }

  CheckCOOQueryAcl(
      gather_status, "COOGetDataAndIndices gather launch");
  scratch.Synchronize();
  scratch.ReleaseChecked();

  return {out_rows, out_cols, out_data};
}

}  // namespace

template <>
std::vector<NDArray>
COOGetDataAndIndices<kDGLAscend, int32_t>(
    COOMatrix coo, NDArray rows, NDArray cols) {
  return COOGetDataAndIndicesAscend<int32_t>(
      coo, rows, cols);
}

template <>
std::vector<NDArray>
COOGetDataAndIndices<kDGLAscend, int64_t>(
    COOMatrix coo, NDArray rows, NDArray cols) {
  return COOGetDataAndIndicesAscend<int64_t>(
      coo, rows, cols);
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_USE_ASCEND