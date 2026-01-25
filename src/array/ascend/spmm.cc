#include <dgl/array.h>
#include <dgl/aten/csr.h>
#include <dgl/runtime/device_api.h>
#include "../kernel_decl.h"
#include <vector>

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }
#endif

namespace dgl {
namespace aten {

/**
 * @brief Ascend implementation of SpMM on CSR format.
 * 
 * Current implementation uses CPU fallback strategy:
 * 1. Synchronize NPU operations
 * 2. Transfer data from NPU to CPU
 * 3. Perform computation on CPU
 * 4. Transfer results back to NPU
 * 
 * @note Only supports copy_lhs + sum operation currently
 */
template <typename IdType, typename DType>
void SpMMCsrAscend(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {

  // Step 1: Synchronize NPU device to ensure all previous operations complete
#ifdef DGL_USE_ASCEND
  if (ufeat->ctx.device_type == kDGLAscend) {
    ASCEND_CALL(aclrtSetDevice(ufeat->ctx.device_id));
    ASCEND_CALL(aclrtSynchronizeDevice());
  }
#endif

  // Step 2: Prepare CPU context and copy data from NPU to CPU
  auto cpu_ctx = DGLContext{kDGLCPU, 0};
  
  // Copy CSR structure
  NDArray indptr_cpu = csr.indptr.CopyTo(cpu_ctx);
  NDArray indices_cpu = csr.indices.CopyTo(cpu_ctx);
  NDArray data_cpu;
  bool has_data = CSRHasData(csr);
  if (has_data) {
    data_cpu = csr.data.CopyTo(cpu_ctx);
  }

  // Copy node features
  NDArray ufeat_cpu = ufeat.CopyTo(cpu_ctx);
  
  // Copy edge features if available
  NDArray efeat_cpu;
  bool has_efeat = !aten::IsNullArray(efeat);
  if (has_efeat) {
    efeat_cpu = efeat.CopyTo(cpu_ctx);
  }
  
  // Step 3: Allocate output buffer on CPU
  auto shape_vec = std::vector<int64_t>(out->shape, out->shape + out->ndim);
  NDArray out_cpu = NDArray::Empty(shape_vec, out->dtype, cpu_ctx);
  
  // Step 4: Extract pointers for computation
  const IdType* indptr_ptr = indptr_cpu.Ptr<IdType>();
  const IdType* indices_ptr = indices_cpu.Ptr<IdType>();
  const IdType* edge_ids_ptr = has_data ? data_cpu.Ptr<IdType>() : nullptr;
  const DType* ufeat_ptr = ufeat_cpu.Ptr<DType>();
  const DType* efeat_ptr = has_efeat ? efeat_cpu.Ptr<DType>() : nullptr;
  DType* out_ptr = out_cpu.Ptr<DType>();
  
  // Get dimensions
  int64_t num_rows = csr.num_rows;
  int64_t out_dim = (out->ndim > 1) ? out->shape[1] : 1;
  int64_t ufeat_dim = (ufeat->ndim > 1) ? ufeat->shape[1] : 1;
  
  // Step 5: Perform SpMM computation
  if (op == "copy_lhs" && reduce == "sum") {
    // Initialize output to zero
    std::memset(out_ptr, 0, num_rows * out_dim * sizeof(DType));
    
    // Main computation loop: out[i] = sum_j (edge_weight[i,j] * ufeat[j])
    for (int64_t i = 0; i < num_rows; ++i) {
      for (IdType idx = indptr_ptr[i]; idx < indptr_ptr[i + 1]; ++idx) {
        IdType col = indices_ptr[idx];  // Neighbor node ID
        IdType eid = has_data ? edge_ids_ptr[idx] : idx;  // Edge ID for indexing efeat
        
        // Get edge weight: from efeat if available, otherwise default to 1.0
        DType edge_weight = has_efeat ? efeat_ptr[eid] : static_cast<DType>(1.0);
        
        // Accumulate weighted neighbor features
        for (int64_t k = 0; k < out_dim; ++k) {
          out_ptr[i * out_dim + k] += edge_weight * ufeat_ptr[col * ufeat_dim + k];
        }
      }
    }
  } else {
    LOG(FATAL) << "SpMMCsrAscend only supports copy_lhs+sum operation. "
               << "Got: op=" << op << ", reduce=" << reduce;
  }

  // Step 6: Copy results back to NPU
  out.CopyFrom(out_cpu);
  
  // Step 7: Synchronize to ensure transfer completion
#ifdef DGL_USE_ASCEND
  if (out->ctx.device_type == kDGLAscend) {
    ASCEND_CALL(aclrtSynchronizeDevice());
  }
#endif
}

template <>
void SpMMCsr<kDGLAscend, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  SpMMCsrAscend<int32_t, float>(op, reduce, bcast, csr, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCsr<kDGLAscend, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  SpMMCsrAscend<int64_t, float>(op, reduce, bcast, csr, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCsr<kDGLAscend, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
    LOG(FATAL) << "Double precision not fully supported on Ascend yet.";
}

template <>
void SpMMCsr<kDGLAscend, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
    LOG(FATAL) << "Double precision not fully supported on Ascend yet.";
}


/**
 * @brief Ascend implementation of SpMM on COO format.
 * 
 * @note Not implemented yet. COO format SpMM operations will fall back to error.
 * @todo Implement COO SpMM with CPU fallback or native Ascend kernels
 */
template <typename IdType, typename DType>
void SpMMCooAscend(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  LOG(FATAL) << "SpMMCoo on Ascend is not implemented yet. "
             << "Op: " << op << ", Reduce: " << reduce;
}

template <>
void SpMMCoo<kDGLAscend, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  SpMMCooAscend<int32_t, float>(op, reduce, bcast, coo, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCoo<kDGLAscend, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  SpMMCooAscend<int64_t, float>(op, reduce, bcast, coo, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCoo<kDGLAscend, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
    LOG(FATAL) << "Double precision not fully supported on Ascend yet.";
}

template <>
void SpMMCoo<kDGLAscend, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
    LOG(FATAL) << "Double precision not fully supported on Ascend yet.";
}

} // namespace aten
} // namespace dgl
