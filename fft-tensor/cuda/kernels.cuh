/**
 * FFT-Tensor CUDA Kernels Header
 * Production-grade sparse spectral tensor operations
 * Optimized for NVIDIA Turing (GTX 1660 Super, compute capability 7.5)
 */

#ifndef FFT_TENSOR_KERNELS_CUH
#define FFT_TENSOR_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cufft.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT error at %s:%d: %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constants
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define SHARED_MEM_SIZE 48 * 1024  // 48KB shared memory per SM on Turing

// Complex number helpers
__device__ __forceinline__ cuFloatComplex complex_mul(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

__device__ __forceinline__ cuFloatComplex complex_add(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float complex_abs(cuFloatComplex a) {
    return sqrtf(a.x * a.x + a.y * a.y);
}

__device__ __forceinline__ cuFloatComplex complex_scale(cuFloatComplex a, float s) {
    return make_cuFloatComplex(a.x * s, a.y * s);
}

// ND indexing helpers
__device__ __forceinline__ int64_t nd_to_linear(
    const int64_t* indices,
    const int64_t* strides,
    int ndim
) {
    int64_t linear_idx = 0;
    for (int i = 0; i < ndim; i++) {
        linear_idx += indices[i] * strides[i];
    }
    return linear_idx;
}

__device__ __forceinline__ void linear_to_nd(
    int64_t linear_idx,
    const int64_t* shape,
    int64_t* indices,
    int ndim
) {
    for (int i = ndim - 1; i >= 0; i--) {
        indices[i] = linear_idx % shape[i];
        linear_idx /= shape[i];
    }
}

// Kernel function declarations
namespace fft_tensor {
namespace kernels {

// Sparse gather/scatter operations
__global__ void sparse_gather_complex_nd(
    const cuFloatComplex* __restrict__ dense_freq,
    const int64_t* __restrict__ sparse_indices,
    cuFloatComplex* __restrict__ sparse_coeffs,
    const int64_t* __restrict__ strides,
    int64_t n_coeffs,
    int ndim
);

__global__ void sparse_scatter_complex_nd(
    cuFloatComplex* __restrict__ dense_freq,
    const int64_t* __restrict__ sparse_indices,
    const cuFloatComplex* __restrict__ sparse_coeffs,
    const int64_t* __restrict__ strides,
    int64_t n_coeffs,
    int ndim
);

// Sparse frequency operations
__global__ void sparse_freq_multiply_complex(
    const cuFloatComplex* __restrict__ coeffs_a,
    const int64_t* __restrict__ indices_a,
    int64_t n_a,
    const cuFloatComplex* __restrict__ coeffs_b,
    const int64_t* __restrict__ indices_b,
    int64_t n_b,
    cuFloatComplex* __restrict__ result_coeffs,
    int64_t* __restrict__ result_indices,
    int64_t* __restrict__ result_count,
    int ndim
);

__global__ void sparse_freq_add_complex(
    const cuFloatComplex* __restrict__ coeffs_a,
    const int64_t* __restrict__ indices_a,
    int64_t n_a,
    const cuFloatComplex* __restrict__ coeffs_b,
    const int64_t* __restrict__ indices_b,
    int64_t n_b,
    cuFloatComplex* __restrict__ result_coeffs,
    int64_t* __restrict__ result_indices,
    int64_t* __restrict__ result_count,
    int ndim
);

// Magnitude and thresholding
__global__ void compute_magnitude(
    const cuFloatComplex* __restrict__ freq_data,
    float* __restrict__ magnitude,
    int64_t n
);

__global__ void threshold_and_compress(
    const cuFloatComplex* __restrict__ freq_data,
    const float* __restrict__ magnitude,
    float threshold,
    cuFloatComplex* __restrict__ sparse_coeffs,
    int64_t* __restrict__ sparse_indices,
    int64_t* __restrict__ count,
    int64_t n
);

// Optimized reductions using shared memory
__global__ void reduce_sum_complex_shared(
    const cuFloatComplex* __restrict__ input,
    cuFloatComplex* __restrict__ output,
    int64_t n
);

// Tensor core operations (for matrix multiplication)
__global__ void tensor_core_matmul_fp16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
);

// Memory management
__global__ void zero_fill(
    cuFloatComplex* __restrict__ data,
    int64_t n
);

__global__ void copy_and_pad(
    const cuFloatComplex* __restrict__ src,
    cuFloatComplex* __restrict__ dst,
    const int64_t* __restrict__ src_shape,
    const int64_t* __restrict__ dst_shape,
    int ndim
);

// Spectral normalization
__global__ void spectral_normalize_inplace(
    cuFloatComplex* __restrict__ coeffs,
    int64_t n,
    float eps
);

// Top-K selection for sparsification
__global__ void topk_threshold_kernel(
    const float* __restrict__ magnitudes,
    float* __restrict__ threshold,
    int64_t n,
    int64_t k
);

} // namespace kernels
} // namespace fft_tensor

#endif // FFT_TENSOR_KERNELS_CUH
