cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fft_tensor_cuda LANGUAGES CXX CUDA)

# Find packages
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architecture for GTX 1660 Super (Turing, 7.5)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Source files
set(CUDA_SOURCES
    fft_ops.cu
    kernels.cu
)

set(CUDA_HEADERS
    kernels.cuh
)

# Create CUDA library
add_library(fft_tensor_cuda MODULE ${CUDA_SOURCES} ${CUDA_HEADERS})

# Include directories
target_include_directories(fft_tensor_cuda PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(fft_tensor_cuda PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cufft
    CUDA::cudart
)

# CUDA flags
set_target_properties(fft_tensor_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Compiler flags
target_compile_options(fft_tensor_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        -lineinfo
        --ptxas-options=-v
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -Wall
    >
)

# Install
install(TARGETS fft_tensor_cuda
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
