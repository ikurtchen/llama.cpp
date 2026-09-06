cmake_minimum_required(VERSION 3.20)

# The SYCL backend is compiled with icpx (set via -DCMAKE_CXX_COMPILER=icpx). Unlike the CUDA
# backend, there is no separate device-compiler frontend to invoke: -fsycl on the normal C++
# compiler handles both host and device code, so this file only needs to add the right flags to
# the ggml-sycl target, not a whole second language.
if (NOT CMAKE_CXX_COMPILER MATCHES "icpx")
    message(WARNING "GGML_SYCL requires icpx as CMAKE_CXX_COMPILER (got: ${CMAKE_CXX_COMPILER})")
endif()

file(GLOB   GGML_HEADERS_SYCL "*.hpp")
file(GLOB   GGML_SOURCES_SYCL "*.cpp")

ggml_add_backend_library(ggml-sycl
                          ${GGML_HEADERS_SYCL}
                          ${GGML_SOURCES_SYCL}
                         )

target_compile_definitions(ggml-sycl PRIVATE GGML_USE_SYCL)

if (GGML_SYCL_F16)
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_F16)
endif()

set(SYCL_COMPILE_FLAGS -fsycl -fsycl-unnamed-lambda -Wno-narrowing)
set(SYCL_LINK_FLAGS    -fsycl)

# AOT compilation for the target device (e.g. bmg-g31) removes first-run JIT latency and is what
# the project's own backend_switch requests. Leaving GGML_SYCL_DEVICE_ARCH empty falls back to
# JIT (portable SPIR-V, picked at runtime) -- useful for shader-dump/stall profiling, which only
# works on a JIT build.
if (GGML_SYCL_DEVICE_ARCH)
    list(APPEND SYCL_COMPILE_FLAGS -fsycl-targets=spir64_gen -Xs "-device ${GGML_SYCL_DEVICE_ARCH}")
    list(APPEND SYCL_LINK_FLAGS    -fsycl-targets=spir64_gen -Xs "-device ${GGML_SYCL_DEVICE_ARCH}")
endif()

target_compile_options(ggml-sycl PRIVATE ${SYCL_COMPILE_FLAGS})
target_link_options   (ggml-sycl PRIVATE ${SYCL_LINK_FLAGS})

# oneMKL (SYCL/DPC++ build, bundled with the oneAPI base toolkit) backs the dense/quantized GEMM
# paths (mul_mat) -- the cuBLAS replacement. Linked unconditionally: every dense model needs mul_mat.
find_package(MKL CONFIG)
if (MKL_FOUND)
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_USE_ONEMKL)
    target_link_libraries(ggml-sycl PRIVATE MKL::MKL_DPCPP)
else()
    message(WARNING "oneMKL (SYCL) not found -- mul_mat will use the hand-written SYCL fallback only, no oneMKL GEMM path")
endif()

if (GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
    find_library(LEVEL_ZERO_LIB NAMES ze_loader)
    if (LEVEL_ZERO_LIB)
        target_link_libraries(ggml-sycl PRIVATE ${LEVEL_ZERO_LIB})
    endif()
endif()
