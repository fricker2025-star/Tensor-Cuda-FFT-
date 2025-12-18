/**
 * FFT-Tensor cuFFT Integration and PyTorch Bindings
 * Production implementation with full ND FFT support and memory management
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernels.cuh"

#include <cufft.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>

using namespace fft_tensor::kernels;

// ========================================
// cuFFT Plan Cache (avoid expensive plan creation)
// ========================================

class CuFFTPlanCache {
private:
    struct PlanKey {
        std::vector<int64_t> shape;
        int batch;
        
        bool operator==(const PlanKey& other) const {
            return shape == other.shape && batch == other.batch;
        }
    };
    
    struct PlanKeyHash {
        std::size_t operator()(const PlanKey& k) const {
            std::size_t hash = k.batch;
            for (auto s : k.shape) {
                hash ^= s + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };
    
    std::unordered_map<PlanKey, cufftHandle, PlanKeyHash> forward_plans;
    std::unordered_map<PlanKey, cufftHandle, PlanKeyHash> inverse_plans;
    
public:
    ~CuFFTPlanCache() {
        for (auto& p : forward_plans) {
            cufftDestroy(p.second);
        }
        for (auto& p : inverse_plans) {
            cufftDestroy(p.second);
        }
    }
    
    cufftHandle get_forward_plan(const std::vector<int64_t>& shape, int batch = 1) {
        PlanKey key{shape, batch};
        
        if (forward_plans.find(key) == forward_plans.end()) {
            // Create new plan
            cufftHandle plan;
            int ndim = shape.size();
            std::vector<int> n(shape.begin(), shape.end());
            
            if (ndim == 1) {
                CUFFT_CHECK(cufftPlan1d(&plan, n[0], CUFFT_C2C, batch));
            } else if (ndim == 2) {
                CUFFT_CHECK(cufftPlan2d(&plan, n[0], n[1], CUFFT_C2C));
            } else if (ndim == 3) {
                CUFFT_CHECK(cufftPlan3d(&plan, n[0], n[1], n[2], CUFFT_C2C));
            } else {
                // ND FFT
                CUFFT_CHECK(cufftPlanMany(&plan, ndim, n.data(),
                                         NULL, 1, 0,
                                         NULL, 1, 0,
                                         CUFFT_C2C, batch));
            }
            
            forward_plans[key] = plan;
        }
        
        return forward_plans[key];
    }
    
    cufftHandle get_inverse_plan(const std::vector<int64_t>& shape, int batch = 1) {
        PlanKey key{shape, batch};
        
        if (inverse_plans.find(key) == inverse_plans.end()) {
            // Inverse plan same as forward for C2C
            inverse_plans[key] = get_forward_plan(shape, batch);
        }
        
        return inverse_plans[key];
    }
    
    void clear() {
        for (auto& p : forward_plans) {
            cufftDestroy(p.second);
        }
        for (auto& p : inverse_plans) {
            cufftDestroy(p.second);
        }
        forward_plans.clear();
        inverse_plans.clear();
    }
};

// Global plan cache
static CuFFTPlanCache g_plan_cache;

// ========================================
// FFT Operations
// ========================================

torch::Tensor fft_forward_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.is_complex(), "Input must be complex tensor");
    
    at::cuda::CUDAGuard device_guard(input.device());
    
    // Get shape
    std::vector<int64_t> shape;
    for (int i = 0; i < input.dim(); i++) {
        shape.push_back(input.size(i));
    }
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Get cuFFT plan
    cufftHandle plan = g_plan_cache.get_forward_plan(shape);
    
    // Execute FFT
    cufftComplex* input_ptr = reinterpret_cast<cufftComplex*>(input.data_ptr<c10::complex<float>>());
    cufftComplex* output_ptr = reinterpret_cast<cufftComplex*>(output.data_ptr<c10::complex<float>>());
    
    CUFFT_CHECK(cufftExecC2C(plan, input_ptr, output_ptr, CUFFT_FORWARD));
    
    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return output;
}

torch::Tensor fft_inverse_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.is_complex(), "Input must be complex tensor");
    
    at::cuda::CUDAGuard device_guard(input.device());
    
    // Get shape
    std::vector<int64_t> shape;
    for (int i = 0; i < input.dim(); i++) {
        shape.push_back(input.size(i));
    }
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Get cuFFT plan
    cufftHandle plan = g_plan_cache.get_inverse_plan(shape);
    
    // Execute inverse FFT
    cufftComplex* input_ptr = reinterpret_cast<cufftComplex*>(input.data_ptr<c10::complex<float>>());
    cufftComplex* output_ptr = reinterpret_cast<cufftComplex*>(output.data_ptr<c10::complex<float>>());
    
    CUFFT_CHECK(cufftExecC2C(plan, input_ptr, output_ptr, CUFFT_INVERSE));
    
    // Normalize by 1/N
    int64_t n = input.numel();
    output = output / static_cast<float>(n);
    
    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return output;
}

// ========================================
// Sparse Operations
// ========================================

std::tuple<torch::Tensor, torch::Tensor> sparse_gather_cuda(
    torch::Tensor dense_freq,
    torch::Tensor indices
) {
    TORCH_CHECK(dense_freq.is_cuda(), "dense_freq must be CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA tensor");
    TORCH_CHECK(dense_freq.is_complex(), "dense_freq must be complex");
    
    at::cuda::CUDAGuard device_guard(dense_freq.device());
    
    int64_t n_coeffs = indices.size(0);
    int ndim = indices.size(1);
    
    // Allocate output
    auto sparse_coeffs = torch::empty({n_coeffs}, 
                                     torch::TensorOptions()
                                         .dtype(dense_freq.dtype())
                                         .device(dense_freq.device()));
    
    // Compute strides
    auto strides = dense_freq.strides();
    auto strides_tensor = torch::tensor(strides, 
                                       torch::TensorOptions()
                                           .dtype(torch::kInt64)
                                           .device(dense_freq.device()));
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (n_coeffs + threads - 1) / threads;
    
    sparse_gather_complex_nd<<<blocks, threads>>>(
        reinterpret_cast<cuFloatComplex*>(dense_freq.data_ptr<c10::complex<float>>()),
        indices.data_ptr<int64_t>(),
        reinterpret_cast<cuFloatComplex*>(sparse_coeffs.data_ptr<c10::complex<float>>()),
        strides_tensor.data_ptr<int64_t>(),
        n_coeffs,
        ndim
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return std::make_tuple(sparse_coeffs, indices);
}

torch::Tensor sparse_scatter_cuda(
    torch::Tensor sparse_coeffs,
    torch::Tensor indices,
    std::vector<int64_t> shape
) {
    TORCH_CHECK(sparse_coeffs.is_cuda(), "sparse_coeffs must be CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA tensor");
    
    at::cuda::CUDAGuard device_guard(sparse_coeffs.device());
    
    int64_t n_coeffs = sparse_coeffs.size(0);
    int ndim = indices.size(1);
    
    // Create dense output (zero-filled)
    auto dense_freq = torch::zeros(shape,
                                   torch::TensorOptions()
                                       .dtype(sparse_coeffs.dtype())
                                       .device(sparse_coeffs.device()));
    
    // Compute strides
    auto strides = dense_freq.strides();
    auto strides_tensor = torch::tensor(strides,
                                       torch::TensorOptions()
                                           .dtype(torch::kInt64)
                                           .device(sparse_coeffs.device()));
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (n_coeffs + threads - 1) / threads;
    
    sparse_scatter_complex_nd<<<blocks, threads>>>(
        reinterpret_cast<cuFloatComplex*>(dense_freq.data_ptr<c10::complex<float>>()),
        indices.data_ptr<int64_t>(),
        reinterpret_cast<cuFloatComplex*>(sparse_coeffs.data_ptr<c10::complex<float>>()),
        strides_tensor.data_ptr<int64_t>(),
        n_coeffs,
        ndim
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return dense_freq;
}

// ========================================
// Sparsification (Top-K)
// ========================================

std::tuple<torch::Tensor, torch::Tensor> sparsify_topk_cuda(
    torch::Tensor freq_data,
    float sparsity
) {
    TORCH_CHECK(freq_data.is_cuda(), "freq_data must be CUDA tensor");
    TORCH_CHECK(freq_data.is_complex(), "freq_data must be complex");
    
    at::cuda::CUDAGuard device_guard(freq_data.device());
    
    int64_t n = freq_data.numel();
    int64_t k = std::max(1L, static_cast<int64_t>(n * sparsity));
    
    // Compute magnitudes
    auto magnitudes = torch::empty({n},
                                   torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(freq_data.device()));
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    compute_magnitude<<<blocks, threads>>>(
        reinterpret_cast<cuFloatComplex*>(freq_data.data_ptr<c10::complex<float>>()),
        magnitudes.data_ptr<float>(),
        n
    );
    
    // Find k-th largest (threshold)
    auto sorted = torch::topk(magnitudes, k);
    float threshold = sorted.values[-1].item<float>();
    
    // Allocate output (worst case: all pass)
    auto sparse_coeffs = torch::empty({k},
                                     torch::TensorOptions()
                                         .dtype(freq_data.dtype())
                                         .device(freq_data.device()));
    auto sparse_indices = torch::empty({k},
                                       torch::TensorOptions()
                                           .dtype(torch::kInt64)
                                           .device(freq_data.device()));
    auto count = torch::zeros({1},
                             torch::TensorOptions()
                                 .dtype(torch::kInt64)
                                 .device(freq_data.device()));
    
    // Threshold and compress
    threshold_and_compress<<<blocks, threads>>>(
        reinterpret_cast<cuFloatComplex*>(freq_data.data_ptr<c10::complex<float>>()),
        magnitudes.data_ptr<float>(),
        threshold,
        reinterpret_cast<cuFloatComplex*>(sparse_coeffs.data_ptr<c10::complex<float>>()),
        sparse_indices.data_ptr<int64_t>(),
        count.data_ptr<int64_t>(),
        n
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Trim to actual count
    int64_t actual_count = count.item<int64_t>();
    sparse_coeffs = sparse_coeffs.slice(0, 0, actual_count);
    sparse_indices = sparse_indices.slice(0, 0, actual_count);
    
    return std::make_tuple(sparse_coeffs, sparse_indices);
}

// ========================================
// Memory Management
// ========================================

void clear_fft_cache() {
    g_plan_cache.clear();
}

int64_t get_cuda_memory_allocated() {
    return c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes[0].current;
}

int64_t get_cuda_memory_reserved() {
    return c10::cuda::CUDACachingAllocator::getDeviceStats(0).reserved_bytes[0].current;
}

void empty_cuda_cache() {
    c10::cuda::CUDACachingAllocator::emptyCache();
}

// ========================================
// PyBind11 Module Definition
// ========================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FFT-Tensor CUDA operations with cuFFT integration";
    
    // FFT operations
    m.def("fft_forward", &fft_forward_cuda, "Forward FFT (CUDA)");
    m.def("fft_inverse", &fft_inverse_cuda, "Inverse FFT (CUDA)");
    
    // Sparse operations
    m.def("sparse_gather", &sparse_gather_cuda, "Sparse gather (CUDA)");
    m.def("sparse_scatter", &sparse_scatter_cuda, "Sparse scatter (CUDA)");
    m.def("sparsify_topk", &sparsify_topk_cuda, "Top-K sparsification (CUDA)");
    
    // Memory management
    m.def("clear_fft_cache", &clear_fft_cache, "Clear cuFFT plan cache");
    m.def("get_cuda_memory_allocated", &get_cuda_memory_allocated, "Get CUDA memory allocated");
    m.def("get_cuda_memory_reserved", &get_cuda_memory_reserved, "Get CUDA memory reserved");
    m.def("empty_cuda_cache", &empty_cuda_cache, "Empty CUDA cache");
}
