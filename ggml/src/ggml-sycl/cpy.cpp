#include "cpy.hpp"

#include <float.h>

#include "dequantize.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml.h"

// Generic scalar copy kernel for type conversions with non-contiguous memory
template <typename src_t, typename dst_t>
static void cpy_scalar_kernel(
        const char * cx,
        char * cdst,
        const int64_t ne,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const sycl::nd_item<1> & item_ct1) {

    const int64_t i = item_ct1.get_global_id(0);

    if (i >= ne) {
        return;
    }

    // Determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int64_t x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int64_t i13 = i / (ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int64_t dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    const src_t * src_val = (const src_t *)(cx + x_offset);
    dst_t * dst_val = (dst_t *)(cdst + dst_offset);

    *dst_val = static_cast<dst_t>(*src_val);
}

// Contiguous copy kernel - simpler and faster when both tensors are contiguous
template <typename src_t, typename dst_t>
static void cpy_contiguous_kernel(
        const src_t * src,
        dst_t * dst,
        const int64_t ne,
        const sycl::nd_item<1> & item_ct1) {

    const int64_t i = item_ct1.get_global_id(0);

    if (i >= ne) {
        return;
    }

    dst[i] = static_cast<dst_t>(src[i]);
}

// Launcher for scalar copy (non-contiguous)
template <typename src_t, typename dst_t>
static void ggml_cpy_scalar_sycl(
        const char * cx,
        char * cdst,
        const int64_t ne,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
        dpct::queue_ptr stream) {

    const int64_t num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks * SYCL_CPY_BLOCK_SIZE), sycl::range<1>(SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            cpy_scalar_kernel<src_t, dst_t>(
                cx, cdst, ne,
                ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                item_ct1);
        });
}

// Launcher for contiguous copy
template <typename src_t, typename dst_t>
static void ggml_cpy_contiguous_sycl(
        const char * cx,
        char * cdst,
        const int64_t ne,
        dpct::queue_ptr stream) {

    const src_t * src = (const src_t *)cx;
    dst_t * dst = (dst_t *)cdst;

    const int64_t num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks * SYCL_CPY_BLOCK_SIZE), sycl::range<1>(SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            cpy_contiguous_kernel<src_t, dst_t>(src, dst, ne, item_ct1);
        });
}

// Block dequantization functions for Q -> F32 copy
static void cpy_blck_q8_0_f32(const char * cxi, char * cdsti) {
    const block_q8_0 * xi = (const block_q8_0 *)cxi;
    float * dsti = (float *)cdsti;

    const float d = xi->d;
    for (int j = 0; j < QK8_0; ++j) {
        dsti[j] = xi->qs[j] * d;
    }
}

static void cpy_blck_q4_0_f32(const char * cxi, char * cdsti) {
    const block_q4_0 * xi = (const block_q4_0 *)cxi;
    float * dsti = (float *)cdsti;

    const float d = xi->d;
    for (int j = 0; j < QK4_0/2; ++j) {
        const int v0 = (xi->qs[j] & 0x0F) - 8;
        const int v1 = (xi->qs[j] >> 4) - 8;
        dsti[j]          = v0 * d;
        dsti[j + QK4_0/2] = v1 * d;
    }
}

static void cpy_blck_q4_1_f32(const char * cxi, char * cdsti) {
    const block_q4_1 * xi = (const block_q4_1 *)cxi;
    float * dsti = (float *)cdsti;

    const float d = xi->dm[0];
    const float m = xi->dm[1];
    for (int j = 0; j < QK4_1/2; ++j) {
        const int v0 = (xi->qs[j] & 0x0F);
        const int v1 = (xi->qs[j] >> 4);
        dsti[j]          = v0 * d + m;
        dsti[j + QK4_1/2] = v1 * d + m;
    }
}

static void cpy_blck_q5_0_f32(const char * cxi, char * cdsti) {
    const block_q5_0 * xi = (const block_q5_0 *)cxi;
    float * dsti = (float *)cdsti;

    const float d = xi->d;
    uint32_t qh;
    memcpy(&qh, xi->qh, sizeof(qh));

    for (int j = 0; j < QK5_0/2; ++j) {
        const int xh_0 = ((qh >> (j + 0))  << 4) & 0x10;
        const int xh_1 = ((qh >> (j + 12))     ) & 0x10;
        const int v0 = ((xi->qs[j] & 0x0F) | xh_0) - 16;
        const int v1 = ((xi->qs[j] >> 4)   | xh_1) - 16;
        dsti[j]          = v0 * d;
        dsti[j + QK5_0/2] = v1 * d;
    }
}

static void cpy_blck_q5_1_f32(const char * cxi, char * cdsti) {
    const block_q5_1 * xi = (const block_q5_1 *)cxi;
    float * dsti = (float *)cdsti;

    const float d = xi->dm[0];
    const float m = xi->dm[1];
    uint32_t qh;
    memcpy(&qh, xi->qh, sizeof(qh));

    for (int j = 0; j < QK5_1/2; ++j) {
        const int xh_0 = ((qh >> (j + 0))  << 4) & 0x10;
        const int xh_1 = ((qh >> (j + 12))     ) & 0x10;
        const int v0 = (xi->qs[j] & 0x0F) | xh_0;
        const int v1 = (xi->qs[j] >> 4)   | xh_1;
        dsti[j]          = v0 * d + m;
        dsti[j + QK5_1/2] = v1 * d + m;
    }
}

// Dequantization copy: Q -> F32
template <cpy_kernel_t cpy_blck, int qk>
static void cpy_q_f32_kernel(
        const char * cx,
        char * cdst,
        const int64_t ne,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const sycl::nd_item<1> & item_ct1) {

    const int64_t i = item_ct1.get_global_id(0) * qk;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int64_t x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int64_t i13 = i / (ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int64_t dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

// Launcher for Q -> F32 copy
template <cpy_kernel_t cpy_blck, int qk>
static void ggml_cpy_q_f32_sycl(
        const char * cx,
        char * cdst,
        const int64_t ne,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
        dpct::queue_ptr stream) {

    GGML_ASSERT(ne % qk == 0);
    const int64_t num_blocks = ne / qk;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks), sycl::range<1>(1)),
        [=](sycl::nd_item<1> item_ct1) {
            cpy_q_f32_kernel<cpy_blck, qk>(
                cx, cdst, ne,
                ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                item_ct1);
        });
}

// Quantization copy: f32 -> q8_0
template <cpy_kernel_t cpy_blck, int qk>
static void cpy_f32_q_kernel(
        const char * cx,
        char * cdst,
        const int64_t ne,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const sycl::nd_item<1> & item_ct1) {

    const int64_t i = item_ct1.get_global_id(0) * qk;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int64_t x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int64_t i13 = i / (ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int64_t dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

// Launcher for f32 -> quantized copy
template <cpy_kernel_t cpy_blck, int qk>
static void ggml_cpy_f32_q_sycl(
        const char * cx,
        char * cdst,
        const int64_t ne,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
        dpct::queue_ptr stream) {

    GGML_ASSERT(ne % qk == 0);
    const int64_t num_blocks = ne / qk;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks), sycl::range<1>(1)),
        [=](sycl::nd_item<1> item_ct1) {
            cpy_f32_q_kernel<cpy_blck, qk>(
                cx, cdst, ne,
                ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                item_ct1);
        });
}

void ggml_sycl_cpy(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1) try {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;

    const bool contiguous_srcs = ggml_is_contiguous(src0) && ggml_is_contiguous(src1);

    // Same type, contiguous: use memcpy
    if (src0->type == src1->type && contiguous_srcs) {
        GGML_ASSERT(ggml_nbytes(src0) == ggml_nbytes(src1));
        main_stream->memcpy(src1_ddc, src0_ddc, ggml_nbytes(src0));
    }
    // F32 -> F32
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_contiguous_sycl<float, float>(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_sycl<float, float>(
                src0_ddc, src1_ddc, ne,
                ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                main_stream);
        }
    }
    // F32 -> F16
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            ggml_cpy_contiguous_sycl<float, sycl::half>(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_sycl<float, sycl::half>(
                src0_ddc, src1_ddc, ne,
                ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                main_stream);
        }
    }
    // F16 -> F32
    else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_contiguous_sycl<sycl::half, float>(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_sycl<sycl::half, float>(
                src0_ddc, src1_ddc, ne,
                ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                main_stream);
        }
    }
    // F16 -> F16
    else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            ggml_cpy_contiguous_sycl<sycl::half, sycl::half>(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_sycl<sycl::half, sycl::half>(
                src0_ddc, src1_ddc, ne,
                ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                main_stream);
        }
    }
    // F32 -> Q8_0
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q_sycl<cpy_blck_f32_q8_0, QK8_0>(
            src0_ddc, src1_ddc, ne,
            ne00, ne01, ne02, nb00, nb01, nb02, nb03,
            ne10, ne11, ne12, nb10, nb11, nb12, nb13,
            main_stream);
    }
    // F32 -> Q4_0
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q_sycl<cpy_blck_f32_q4_0, QK4_0>(
            src0_ddc, src1_ddc, ne,
            ne00, ne01, ne02, nb00, nb01, nb02, nb03,
            ne10, ne11, ne12, nb10, nb11, nb12, nb13,
            main_stream);
    }
    // F32 -> Q4_1
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q_sycl<cpy_blck_f32_q4_1, QK4_1>(
            src0_ddc, src1_ddc, ne,
            ne00, ne01, ne02, nb00, nb01, nb02, nb03,
            ne10, ne11, ne12, nb10, nb11, nb12, nb13,
            main_stream);
    }
    // F32 -> Q5_0
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q_sycl<cpy_blck_f32_q5_0, QK5_0>(
            src0_ddc, src1_ddc, ne,
            ne00, ne01, ne02, nb00, nb01, nb02, nb03,
            ne10, ne11, ne12, nb10, nb11, nb12, nb13,
            main_stream);
    }
    // F32 -> Q5_1
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q_sycl<cpy_blck_f32_q5_1, QK5_1>(
            src0_ddc, src1_ddc, ne,
            ne00, ne01, ne02, nb00, nb01, nb02, nb03,
            ne10, ne11, ne12, nb10, nb11, nb12, nb13,
            main_stream);
    }
    // F32 -> IQ4_NL
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_q_sycl<cpy_blck_f32_iq4_nl, QK4_NL>(
            src0_ddc, src1_ddc, ne,
            ne00, ne01, ne02, nb00, nb01, nb02, nb03,
            ne10, ne11, ne12, nb10, nb11, nb12, nb13,
            main_stream);
    }
    // Q8_0 -> F32
    else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q_f32_sycl<cpy_blck_q8_0_f32, QK8_0>(
            src0_ddc, src1_ddc, ne,
            ne00, ne01, ne02, nb00, nb01, nb02, nb03,
            ne10, ne11, ne12, nb10, nb11, nb12, nb13,
            main_stream);
    }
    // Q4_0 -> F32
    else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q_f32_sycl<cpy_blck_q4_0_f32, QK4_0>(
            src0_ddc, src1_ddc, ne,
            ne00, ne01, ne02, nb00, nb01, nb02, nb03,
            ne10, ne11, ne12, nb10, nb11, nb12, nb13,
            main_stream);
    }
    // Q4_1 -> F32
    else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q_f32_sycl<cpy_blck_q4_1_f32, QK4_1>(
            src0_ddc, src1_ddc, ne,
            ne00, ne01, ne02, nb00, nb01, nb02, nb03,
            ne10, ne11, ne12, nb10, nb11, nb12, nb13,
            main_stream);
    }
    // Q5_0 -> F32
    else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q_f32_sycl<cpy_blck_q5_0_f32, QK5_0>(
            src0_ddc, src1_ddc, ne,
            ne00, ne01, ne02, nb00, nb01, nb02, nb03,
            ne10, ne11, ne12, nb10, nb11, nb12, nb13,
            main_stream);
    }
    // Q5_1 -> F32
    else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q_f32_sycl<cpy_blck_q5_1_f32, QK5_1>(
            src0_ddc, src1_ddc, ne,
            ne00, ne01, ne02, nb00, nb01, nb02, nb03,
            ne10, ne11, ne12, nb10, nb11, nb12, nb13,
            main_stream);
    }
    else {
        GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__,
                ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_cpy(ctx, dst->src[0], dst);
}
