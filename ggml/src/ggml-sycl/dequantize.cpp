#include "dequantize.hpp"

#include <sycl/sycl.hpp>

namespace {

constexpr int64_t DEQUANTIZE_BLOCK_SIZE = 256;

template <typename dst_t, typename src_t>
void convert_unary_sycl(const void * vx, dst_t * y, int64_t k, sycl::queue & q) {
    const src_t * x = (const src_t *) vx;
    const int64_t nblk = (k + DEQUANTIZE_BLOCK_SIZE - 1) / DEQUANTIZE_BLOCK_SIZE;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * DEQUANTIZE_BLOCK_SIZE), sycl::range<1>(DEQUANTIZE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) {
        const int64_t i = item.get_global_id(0);
        if (i >= k) {
            return;
        }
        y[i] = (dst_t) x[i];
    });
}

template <typename dst_t, typename block_t, dst_t (*dequantize_at)(const block_t * x, int64_t i), int qk>
void dequantize_row_sycl(const void * vx, dst_t * y, int64_t k, sycl::queue & q) {
    GGML_ASSERT(k % qk == 0);
    const block_t * x = (const block_t *) vx;
    const int64_t nblk = (k + DEQUANTIZE_BLOCK_SIZE - 1) / DEQUANTIZE_BLOCK_SIZE;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * DEQUANTIZE_BLOCK_SIZE), sycl::range<1>(DEQUANTIZE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) {
        const int64_t i = item.get_global_id(0);
        if (i >= k) {
            return;
        }
        y[i] = dequantize_at(x, i);
    });
}

} // namespace

void ggml_sycl_convert_to_fp32(sycl::queue & q, const ggml_tensor * src, float * dst) {
    GGML_ASSERT(ggml_is_contiguous(src));
    const to_fp32_sycl_t fn = ggml_get_to_fp32_sycl(src->type);
    GGML_ASSERT(fn != nullptr);
    fn(src->data, dst, ggml_nelements(src), q);
}

void ggml_sycl_convert_to_fp16(sycl::queue & q, const ggml_tensor * src, sycl::half * dst) {
    GGML_ASSERT(ggml_is_contiguous(src));
    const to_fp16_sycl_t fn = ggml_get_to_fp16_sycl(src->type);
    GGML_ASSERT(fn != nullptr);
    fn(src->data, dst, ggml_nelements(src), q);
}

to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_sycl<float, float>;
        case GGML_TYPE_F16:
            return convert_unary_sycl<float, sycl::half>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_sycl<float, block_q4_0, ggml_sycl_dequantize_q4_0_at<float>, QK4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_row_sycl<float, block_q4_1, ggml_sycl_dequantize_q4_1_at<float>, QK4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_row_sycl<float, block_q5_0, ggml_sycl_dequantize_q5_0_at<float>, QK5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_row_sycl<float, block_q5_1, ggml_sycl_dequantize_q5_1_at<float>, QK5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_row_sycl<float, block_q8_0, ggml_sycl_dequantize_q8_0_at<float>, QK8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_sycl<float, block_q2_K, ggml_sycl_dequantize_q2_K_at<float>, QK_K>;
        case GGML_TYPE_Q3_K:
            return dequantize_row_sycl<float, block_q3_K, ggml_sycl_dequantize_q3_K_at<float>, QK_K>;
        case GGML_TYPE_Q4_K:
            return dequantize_row_sycl<float, block_q4_K, ggml_sycl_dequantize_q4_K_at<float>, QK_K>;
        case GGML_TYPE_Q5_K:
            return dequantize_row_sycl<float, block_q5_K, ggml_sycl_dequantize_q5_K_at<float>, QK_K>;
        case GGML_TYPE_Q6_K:
            return dequantize_row_sycl<float, block_q6_K, ggml_sycl_dequantize_q6_K_at<float>, QK_K>;
        default:
            return nullptr;
    }
}

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_sycl<sycl::half, float>;
        case GGML_TYPE_F16:
            return convert_unary_sycl<sycl::half, sycl::half>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_sycl<sycl::half, block_q4_0, ggml_sycl_dequantize_q4_0_at<sycl::half>, QK4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_row_sycl<sycl::half, block_q4_1, ggml_sycl_dequantize_q4_1_at<sycl::half>, QK4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_row_sycl<sycl::half, block_q5_0, ggml_sycl_dequantize_q5_0_at<sycl::half>, QK5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_row_sycl<sycl::half, block_q5_1, ggml_sycl_dequantize_q5_1_at<sycl::half>, QK5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_row_sycl<sycl::half, block_q8_0, ggml_sycl_dequantize_q8_0_at<sycl::half>, QK8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_sycl<sycl::half, block_q2_K, ggml_sycl_dequantize_q2_K_at<sycl::half>, QK_K>;
        case GGML_TYPE_Q3_K:
            return dequantize_row_sycl<sycl::half, block_q3_K, ggml_sycl_dequantize_q3_K_at<sycl::half>, QK_K>;
        case GGML_TYPE_Q4_K:
            return dequantize_row_sycl<sycl::half, block_q4_K, ggml_sycl_dequantize_q4_K_at<sycl::half>, QK_K>;
        case GGML_TYPE_Q5_K:
            return dequantize_row_sycl<sycl::half, block_q5_K, ggml_sycl_dequantize_q5_K_at<sycl::half>, QK_K>;
        case GGML_TYPE_Q6_K:
            return dequantize_row_sycl<sycl::half, block_q6_K, ggml_sycl_dequantize_q6_K_at<sycl::half>, QK_K>;
        default:
            return nullptr;
    }
}
