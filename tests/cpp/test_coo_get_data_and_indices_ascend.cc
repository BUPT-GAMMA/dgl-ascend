#ifdef DGL_USE_ASCEND

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "../../src/array/array_op.h"
#include "./common.h"

namespace {

using dgl::aten::COOMatrix;
using dgl::runtime::NDArray;

template <typename IdType>
struct COOQueryCase {
  std::string name;
  int64_t num_rows;
  int64_t num_cols;
  std::vector<IdType> coo_rows;
  std::vector<IdType> coo_cols;
  std::vector<IdType> coo_data;
  bool has_data;
  std::vector<IdType> rows;
  std::vector<IdType> cols;
};

template <typename IdType>
void AppendCase(
    std::vector<COOQueryCase<IdType>>* cases, const std::string& name,
    int64_t num_rows, int64_t num_cols,
    const std::vector<IdType>& coo_rows,
    const std::vector<IdType>& coo_cols,
    const std::vector<IdType>& coo_data, bool has_data,
    const std::vector<IdType>& rows, const std::vector<IdType>& cols) {
  cases->push_back({name, num_rows, num_cols, coo_rows, coo_cols, coo_data,
                    has_data, rows, cols});
}

template <typename IdType>
std::vector<COOQueryCase<IdType>> MakeCases() {
  std::vector<COOQueryCase<IdType>> cases;
  // 在同一批数据上检查 CPU 两个分支及重复边顺序。
  for (int count : {1, 199, 200, 201}) {
    for (bool hasData : {true, false}) {
      const std::string name =
        std::string("duplicates_") +
        (hasData ? "explicit_q" : "implicit_q") +
        std::to_string(count);

      AppendCase<IdType>(
        &cases, name, 3, 3,
        {1, 0, 1, 1}, {2, 1, 2, 2},
        hasData ? std::vector<IdType>{10, 20, 30, 40}
            : std::vector<IdType>{},
        hasData,
        std::vector<IdType>(count, IdType(1)), {2});
    }
  }

  AppendCase<IdType>(
    &cases, "multiple_queries", 4, 4,
    {1, 0, 1, 2, 1}, {2, 1, 2, 0, 3},
    {10, 20, 30, 40, 50}, true,
    {1, 0, 1, 2}, {2, 1, 3, 0});

  AppendCase<IdType>(
    &cases, "rows_broadcast", 4, 4,
    {1, 1, 1, 0}, {2, 3, 2, 0},
    {}, false, {1}, {2, 3, 0});

  AppendCase<IdType>(
    &cases, "cols_broadcast", 4, 4,
    {1, 0, 1, 2}, {2, 2, 2, 0},
    {10, 20, 30, 40}, true, {1, 0, 2}, {2});

  AppendCase<IdType>(
    &cases, "no_match", 3, 3,
    {1, 1}, {2, 2}, {}, false, {0}, {0});

  AppendCase<IdType>(
    &cases, "empty_coo", 3, 3,
    {}, {}, {}, false, {1}, {2});

  AppendCase<IdType>(
    &cases, "empty_queries", 3, 3,
    {1}, {2}, {}, false, {}, {});

  AppendCase<IdType>(
    &cases, "empty_rows", 3, 3,
    {1}, {2}, {}, false, {}, {2});

  AppendCase<IdType>(
    &cases, "empty_cols", 3, 3,
    {1}, {2}, {}, false, {1}, {});


  // Repeated coordinates span three tiles; several keys share PairHash.
  // Check paired queries and both broadcast directions around threshold 200.
  {
    std::vector<IdType> cooRows(513), cooCols(513), data(513);
    for (size_t k = 0; k < 513; ++k) {
      cooRows[k] = static_cast<IdType>(k % 7);
      cooCols[k] = static_cast<IdType>((k / 7) % 5);
      data[k] = static_cast<IdType>(1000 + k);
    }

    for (int count : {199, 200, 201}) {
      for (bool hasData : {true, false}) {
        for (int mode = 0; mode < 3; ++mode) {
          std::vector<IdType> rows(count), cols(count);
          for (int q = 0; q < count; ++q) {
            rows[q] = static_cast<IdType>(q % 7);
            cols[q] = static_cast<IdType>((q / 7) % 5);
          }
          if (mode == 1) rows.assign(1, IdType(2));
          if (mode == 2) cols.assign(1, IdType(3));

          const std::string name =
            "cross_tile_q" + std::to_string(count) +
            (hasData ? "_explicit_" : "_implicit_") +
            (mode == 0 ? "paired" :
            mode == 1 ? "rows_broadcast" : "cols_broadcast");

          AppendCase<IdType>(
            &cases, name, 7, 5, cooRows, cooCols,
            hasData ? data : std::vector<IdType>{},
            hasData, rows, cols);
        }
      }
    }
  }

  for (int count : {255, 256, 257}) {
    std::vector<IdType> cooRows(count, IdType(0));
    std::vector<IdType> cooCols(count, IdType(0));
    cooRows.front() = cooRows.back() = IdType(1);
    cooCols.front() = cooCols.back() = IdType(2);

    AppendCase<IdType>(
      &cases, "tile_" + std::to_string(count), 3, 3,
      cooRows, cooCols, {}, false,
      {0, 1, 2}, {0, 2, 1});
  }

  // 输入补零区满足查询，但所有有效 COO 元素均不匹配。
  AppendCase<IdType>(
    &cases, "padding_no_match", 3, 3,
    std::vector<IdType>(255, IdType(1)),
    std::vector<IdType>(255, IdType(2)),
    {}, false, {0}, {0});

  // int32 超过 2^24；int64 额外超过 2^32。
  const int64_t high64 =
    (sizeof(IdType) == 8 ? (INT64_C(1) << 40)
              : (INT64_C(1) << 25)) + 1;
  const IdType high = static_cast<IdType>(high64);

  AppendCase<IdType>(
    &cases, "large_coordinate_exact", high64 + 2, high64 + 2,
    {high, IdType(1), IdType(high + 1)},
    {high, high, IdType(high + 1)},
    {}, false,
    {high, IdType(high + 1)},
    {high, IdType(high + 1)});

  AppendCase<IdType>(
    &cases, "signed_data_limits", 3, 3,
    {1, 1}, {2, 2},
    {std::numeric_limits<IdType>::min(),
    std::numeric_limits<IdType>::max()},
    true, {1}, {2});
  return cases;
}

template <typename IdType>
COOMatrix MakeCPUCOO(const COOQueryCase<IdType>& input) {
  return COOMatrix{
      input.num_rows, input.num_cols, NDArray::FromVector(input.coo_rows),
      NDArray::FromVector(input.coo_cols),
      input.has_data ? NDArray::FromVector(input.coo_data)
                     : dgl::aten::NullArray()};
}

template <typename IdType>
void CheckOutputs(
    const std::vector<NDArray>& actual,
    const std::vector<NDArray>& expected, DGLContext ctx) {
  ASSERT_EQ(actual.size(), 3U);
  ASSERT_EQ(expected.size(), 3U);
  const char* names[] = {"ret_rows", "ret_cols", "ret_data"};
  for (size_t i = 0; i < 3; ++i) {
    SCOPED_TRACE(names[i]);
    ASSERT_EQ(actual[i]->ndim, 1);
    EXPECT_EQ(actual[i]->ctx.device_type, ctx.device_type);
    EXPECT_EQ(actual[i]->ctx.device_id, ctx.device_id);
    EXPECT_EQ(actual[i]->dtype.code, kDGLInt);
    EXPECT_EQ(actual[i]->dtype.bits, sizeof(IdType) * 8);
    EXPECT_EQ(actual[i]->dtype.lanes, 1);
    ASSERT_EQ(actual[i]->shape[0], expected[i]->shape[0]);
    // ArrayEQ compares every element, without sorting or removing duplicates.
    EXPECT_TRUE(ArrayEQ<IdType>(actual[i].CopyTo(CPU), expected[i]));
  }
}

template <typename IdType>
void CheckCase(const COOQueryCase<IdType>& input) {
  SCOPED_TRACE(input.name);
  auto cpu_coo = MakeCPUCOO(input);
  auto cpu_rows = NDArray::FromVector(input.rows);
  auto cpu_cols = NDArray::FromVector(input.cols);
  auto expected = dgl::aten::COOGetDataAndIndices(cpu_coo, cpu_rows, cpu_cols);

  auto npu_coo = cpu_coo.CopyTo(NPU);
  auto npu_rows = cpu_rows.CopyTo(NPU);
  auto npu_cols = cpu_cols.CopyTo(NPU);
  auto actual = dgl::aten::COOGetDataAndIndices(npu_coo, npu_rows, npu_cols);
  CheckOutputs<IdType>(actual, expected, NPU);

  // Exercise the backend directly as well, so the previous public CPU
  // fallback cannot conceal a missing or broken Ascend specialization.
  auto backend = dgl::aten::impl::COOGetDataAndIndices<kDGLAscend, IdType>(
      npu_coo, npu_rows, npu_cols);
  CheckOutputs<IdType>(backend, expected, NPU);
}

template <typename IdType>
void CheckAllCases() {
  const auto cases = MakeCases<IdType>();
  ASSERT_EQ(cases.size(), 40U);
  for (const auto& input : cases) {
    CheckCase(input);
    if (::testing::Test::HasFatalFailure()) return;
  }
}

class COOGetDataAndIndicesAscendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto* device = dgl::runtime::DeviceAPI::Get(NPU);
    if (!device->IsAvailable()) {
      GTEST_SKIP() << "Ascend device is unavailable; this is not a passing NPU run";
    }
    device->SetDevice(NPU);
  }
};

TEST_F(COOGetDataAndIndicesAscendTest, Int32MatchesNativeCPU) {
  CheckAllCases<int32_t>();
}

TEST_F(COOGetDataAndIndicesAscendTest, Int64MatchesNativeCPU) {
  CheckAllCases<int64_t>();
}

template <typename IdType>
void CheckInvalidInputs() {
  // Forking a process after ACL initialization is unsafe. The repository's
  // default dmlc throwing logger permits in-process validation instead.
  ASSERT_TRUE(DMLC_LOG_FATAL_THROW)
      << "Invalid-input tests require DMLC_LOG_FATAL_THROW=1";
  auto coo = COOMatrix{
      3, 3, NDArray::FromVector(std::vector<IdType>{1, 1}),
      NDArray::FromVector(std::vector<IdType>{2, 2}),
      NDArray::FromVector(std::vector<IdType>{10, 20})}.CopyTo(NPU);
  const auto npu_array = [](std::vector<IdType> values) {
    return NDArray::FromVector(values).CopyTo(NPU);
  };
  const auto rows = npu_array({1});
  const auto cols = npu_array({2});

  EXPECT_THROW(
      dgl::aten::COOGetDataAndIndices(
          coo, npu_array({1, 1}), npu_array({2, 2, 2})), dmlc::Error);
  EXPECT_THROW(
      dgl::aten::COOGetDataAndIndices(coo, npu_array({-1}), cols),
      dmlc::Error);
  EXPECT_THROW(
      dgl::aten::COOGetDataAndIndices(coo, npu_array({3}), cols),
      dmlc::Error);
  EXPECT_THROW(
      dgl::aten::COOGetDataAndIndices(coo, rows, npu_array({3})),
      dmlc::Error);

  auto bad_coo = coo;
  bad_coo.col = npu_array({2});
  EXPECT_THROW(
      dgl::aten::COOGetDataAndIndices(bad_coo, rows, cols), dmlc::Error);
  bad_coo = coo;
  bad_coo.data = npu_array({10});
  EXPECT_THROW(
      dgl::aten::COOGetDataAndIndices(bad_coo, rows, cols), dmlc::Error);

  using OtherType = typename std::conditional<
      sizeof(IdType) == 4, int64_t, int32_t>::type;
  const auto wrong_type = NDArray::FromVector(
      std::vector<OtherType>{1}).CopyTo(NPU);
  EXPECT_THROW(
      dgl::aten::COOGetDataAndIndices(coo, wrong_type, cols), dmlc::Error);
  EXPECT_THROW(
      dgl::aten::COOGetDataAndIndices(
          coo, NDArray::FromVector(std::vector<IdType>{1}), cols),
      dmlc::Error);

  auto rank_two = NDArray::Empty({1, 1}, rows->dtype, NPU);
  EXPECT_THROW(
      dgl::aten::COOGetDataAndIndices(coo, rank_two, cols), dmlc::Error);

  auto empty = COOMatrix{
      3, 3, NDArray::FromVector(std::vector<IdType>{}),
      NDArray::FromVector(std::vector<IdType>{})}.CopyTo(NPU);
  EXPECT_THROW(
      dgl::aten::COOGetDataAndIndices(empty, npu_array({3}), cols),
      dmlc::Error);

  // A rejected invocation must not reset or finalize the DGL device.
  CheckCase(MakeCases<IdType>().front());
}

TEST_F(COOGetDataAndIndicesAscendTest, RejectsInvalidInt32Inputs) {
  CheckInvalidInputs<int32_t>();
}

TEST_F(COOGetDataAndIndicesAscendTest, RejectsInvalidInt64Inputs) {
  CheckInvalidInputs<int64_t>();
}

class ScopedCOOTestStream {
 public:
  ScopedCOOTestStream()
      : device_(dgl::runtime::DeviceAPI::Get(NPU)),
        previous_(device_->GetStream()), stream_(device_->CreateStream(NPU)) {
    try {
      device_->SetStream(NPU, stream_);
    } catch (...) {
      device_->FreeStream(NPU, stream_);
      throw;
    }
  }

  ~ScopedCOOTestStream() {
    try {
      device_->SetStream(NPU, previous_);
      device_->FreeStream(NPU, stream_);
    } catch (const std::exception& error) {
      ADD_FAILURE() << "Stream cleanup failed: " << error.what();
    }
  }

 private:
  dgl::runtime::DeviceAPI* device_;
  DGLStreamHandle previous_;
  DGLStreamHandle stream_;
};

TEST_F(COOGetDataAndIndicesAscendTest, PreservesCallerStream) {
  auto* device = dgl::runtime::DeviceAPI::Get(NPU);
  const auto previous = device->GetStream();
  {
    ScopedCOOTestStream stream;
    CheckCase(MakeCases<int32_t>()[4]);
    CheckCase(MakeCases<int64_t>()[4]);
    device->StreamSync(NPU, device->GetStream());
  }
  EXPECT_EQ(device->GetStream(), previous);
  CheckCase(MakeCases<int32_t>().front());
}

}  // namespace

#endif  // DGL_USE_ASCEND
