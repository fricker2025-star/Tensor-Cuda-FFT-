/*
CUDA kernels for sparse FFT operations.
Optimized for 1660 Super (Turing architecture, compute capability 7.5)
*/

#include <cuda_runtime.h>
#include <cufft.h>
#include <torch/extension.h>

// Sparse coefficient gather kernel
__global__ void sparse_gather_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> dense_freq,
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> sparse_coeffs,
    int n_coeffs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_coeffs) {
        int64_t i = indices[idx][0];
        int64_t j = indices[idx][1];
        sparse_coeffs[idx] = dense_freq[i][j];
    }
}

// Sparse coefficient scatter kernel
__global__ void sparse_scatter_kernel(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> dense_freq,
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> indices,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> sparse_coeffs,
    int n_coeffs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_coeffs) {
        int64_t i = indices[idx][0];
        int64_t j = indices[idx][1];
        dense_freq[i][j] = sparse_coeffs[idx];
    }
}

// Sparse frequency multiply (for convolution)
__global__ void sparse_freq_multiply_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> coeffs_a,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> coeffs_b,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> result,
    int n_coeffs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_coeffs) {
        result[idx] = coeffs_a[idx] * coeffs_b[idx];
    }
}

// Memory-efficient sparse FFT gather
torch::Tensor sparse_gather_cuda(
    torch::Tensor dense_freq,
    torch::Tensor indices
) {
    const int n_coeffs = indices.size(0);
    const int threads = 256;
    const int blocks = (n_coeffs + threads - 1) / threads;
    
    auto sparse_coeffs = torch::zeros({n_coeffs}, dense_freq.options());
    
    sparse_gather_kernel<<<blocks, threads>>>(
        dense_freq.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        sparse_coeffs.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        n_coeffs
    );
    
    return sparse_coeffs;
}

// Memory-efficient sparse FFT scatter
torch::Tensor sparse_scatter_cuda(
    torch::Tensor sparse_coeffs,
    torch::Tensor indices,
    std::vector<int64_t> shape
) {
    const int n_coeffs = indices.size(0);
    const int threads = 256;
    const int blocks = (n_coeffs + threads - 1) / threads;
    
    auto dense_freq = torch::zeros(shape, sparse_coeffs.options());
    
    sparse_scatter_kernel<<<blocks, threads>>>(
        dense_freq.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        sparse_coeffs.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        n_coeffs
    );
    
    return dense_freq;
}

// Sparse frequency-domain multiply
torch::Tensor sparse_freq_multiply_cuda(
    torch::Tensor coeffs_a,
    torch::Tensor coeffs_b
) {
    const int n_coeffs = coeffs_a.size(0);
    const int threads = 256;
    const int blocks = (n_coeffs + threads - 1) / threads;
    
    auto result = torch::zeros_like(coeffs_a);
    
    sparse_freq_multiply_kernel<<<blocks, threads>>>(
        coeffs_a.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        coeffs_b.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        result.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        n_coeffs
    );
    
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_gather", &sparse_gather_cuda, "Sparse FFT gather (CUDA)");
    m.def("sparse_scatter", &sparse_scatter_cuda, "Sparse FFT scatter (CUDA)");
    m.def("sparse_freq_multiply", &sparse_freq_multiply_cuda, "Sparse frequency multiply (CUDA)");
}
