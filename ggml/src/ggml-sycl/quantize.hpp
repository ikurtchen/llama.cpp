#pragma once

#include "common.hpp"

#define GGML_COMMON_DECL_SYCL
#include "ggml-common.h"

#include <cstdint>

typedef void (*quantize_row_sycl_t)(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, sycl::queue & q);

void ggml_sycl_quantize_row_q8_0(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, sycl::queue & q);

void ggml_sycl_quantize_row_q8_1(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, sycl::queue & q);

inline void ggml_sycl_quantize_f32_to_q8_0(sycl::queue & q, const float * x, block_q8_0 * y, int64_t k) {
    ggml_sycl_quantize_row_q8_0(x, nullptr, y, GGML_TYPE_Q8_0, k, k, k, k, k, 1, 1, 1, q);
}

inline void ggml_sycl_quantize_f32_to_q8_1(sycl::queue & q, const float * x, block_q8_1 * y, int64_t k) {
    ggml_sycl_quantize_row_q8_1(x, nullptr, y, GGML_TYPE_Q8_1, k, k, k, k, k, 1, 1, 1, q);
}
