#include "set_rows.hpp"

#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #include <sycl/ext/oneapi/bfloat16.hpp>
#endif
#include <cstring>
#include <cfloat>

using sycl_bf16 = sycl::ext::oneapi::bfloat16;

// Helper for IQ4_NL quantization: find closest value in a sorted int8 table
static inline int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

// Device-side quantization functions ported from CUDA cpy-utils.cuh

static inline void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < sycl::fabs(v)) {
            amax = sycl::fabs(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = sycl::half(d);

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = sycl::min((int)15, (int)(int8_t)(x0 + 8.5f));
        const uint8_t xi1 = sycl::min((int)15, (int)(int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static inline void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->data.d = sycl::half(d);
    y->data.m = sycl::half(vmin);

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = sycl::min((int)15, (int)(int8_t)(x0 + 0.5f));
        const uint8_t xi1 = sycl::min((int)15, (int)(int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static inline void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < sycl::fabs(v)) {
            amax = sycl::fabs(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = sycl::half(d);

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = sycl::min((int)31, (int)(int8_t)(x0 + 16.5f));
        const uint8_t xi1 = sycl::min((int)31, (int)(int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static inline void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float vmin = x[0];
    float vmax = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        vmin = v < vmin ? v : vmin;
        vmax = v > vmax ? v : vmax;
    }

    const float d  = (vmax - vmin) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->data.d = sycl::half(d);
    y->data.m = sycl::half(vmin);

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK5_1/2 + j] - vmin)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static inline void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f;

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = sycl::fmax(amax, sycl::fabs(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = sycl::half(d);

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = sycl::round(x0);
    }
}

static inline void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < sycl::fabs(v)) {
            amax = sycl::fabs(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sycl::half(sumq2 > 0 ? sumqx/sumq2 : d);
}

// Kernel for quantized dst types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL)
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static void k_set_rows_quant(const float * __restrict__ src0,
                              const idx_t * __restrict__ src1,
                              block_type * __restrict__ dst,
                              const int64_t ne_total,
                              const int64_t s01, const int64_t s02, const int64_t s03,
                              const int64_t s10, const int64_t s11, const int64_t s12,
                              const int64_t s1, const int64_t s2, const int64_t s3,
                              const sycl::uint3 ne00_fd,
                              const sycl::uint3 ne01_fd,
                              const sycl::uint3 ne02_fd,
                              const sycl::uint3 ne11_fd,
                              const sycl::uint3 ne12_fd,
                              const sycl::nd_item<1> & item) {
    const int64_t i = item.get_global_id(0);
    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t tmp = (uint32_t) i_base;
    sycl::uint2 div_mod;

    div_mod           = fast_div_modulo(tmp, ne00_fd);
    const int64_t i00 = div_mod.y();
    tmp               = div_mod.x();

    div_mod           = fast_div_modulo(tmp, ne01_fd);
    const int64_t i01 = div_mod.y();
    tmp               = div_mod.x();

    div_mod           = fast_div_modulo(tmp, ne02_fd);
    const int64_t i02 = div_mod.y();
    const int64_t i03 = div_mod.x();

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    // dst strides s1, s2, s3 are in bytes for quantized types
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);
}

// Launch helper for quantized dst types
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static void set_rows_sycl_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne11, const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        sycl::queue & stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    if (ne_total <= 0 || ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || ne11 <= 0 || ne12 <= 0) {
        return;
    }

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);
    // For quantized dst, strides are kept in bytes
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    const sycl::uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
    const sycl::uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
    const sycl::uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const sycl::uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
    const sycl::uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

    const int num_blocks = (ne_total + SYCL_SET_BLOCK_SIZE - 1) / SYCL_SET_BLOCK_SIZE;

    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_SET_BLOCK_SIZE, SYCL_SET_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            k_set_rows_quant<idx_t, block_type, qk, quantize_func>(
                src0_d, src1_d, dst_d, ne_total,
                s01, s02, s03,
                s10, s11, s12,
                s1, s2, s3,
                ne00_fd, ne01_fd, ne02_fd,
                ne11_fd, ne12_fd, item);
        });
}

// Kernel for float dst types (F32, F16, BF16)
template <typename src_t, typename idx_t, typename dst_t>
static void k_set_rows(const src_t * __restrict__ src0,
                       const idx_t * __restrict__ src1,
                       dst_t * __restrict__ dst,
                       const int64_t ne_total,
                       const int64_t s01, const int64_t s02, const int64_t s03,
                       const int64_t s10, const int64_t s11, const int64_t s12,
                       const int64_t s1, const int64_t s2, const int64_t s3,
                       const sycl::uint3 ne00_fd,
                       const sycl::uint3 ne01_fd,
                       const sycl::uint3 ne02_fd,
                       const sycl::uint3 ne11_fd,
                       const sycl::uint3 ne12_fd,
                       const sycl::nd_item<1> & item) {
    const int64_t i = item.get_global_id(0);
    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    sycl::uint2 div_mod;

    div_mod           = fast_div_modulo(tmp, ne00_fd);
    const int64_t i00 = div_mod.y();
    tmp               = div_mod.x();

    div_mod           = fast_div_modulo(tmp, ne01_fd);
    const int64_t i01 = div_mod.y();
    tmp               = div_mod.x();

    div_mod           = fast_div_modulo(tmp, ne02_fd);
    const int64_t i02 = div_mod.y();
    const int64_t i03 = div_mod.x();

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = (dst_t) src0_row[i00];
}

// Launch helper for a specific (src_t, idx_t, dst_t) combination
template <typename src_t, typename idx_t, typename dst_t>
static void set_rows_sycl(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne11, const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        sycl::queue & stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    if (ne_total <= 0 || ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || ne11 <= 0 || ne12 <= 0) {
        return;
    }

    const int64_t s01 = nb01 / sizeof(src_t);
    const int64_t s02 = nb02 / sizeof(src_t);
    const int64_t s03 = nb03 / sizeof(src_t);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);
    const int64_t s1  = nb1  / sizeof(dst_t);
    const int64_t s2  = nb2  / sizeof(dst_t);
    const int64_t s3  = nb3  / sizeof(dst_t);

    const sycl::uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
    const sycl::uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
    const sycl::uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const sycl::uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
    const sycl::uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

    const int num_blocks = (ne_total + SYCL_SET_BLOCK_SIZE - 1) / SYCL_SET_BLOCK_SIZE;

    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_SET_BLOCK_SIZE, SYCL_SET_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            k_set_rows(src0_d, src1_d, dst_d, ne_total,
                       s01, s02, s03,
                       s10, s11, s12,
                       s1, s2, s3,
                       ne00_fd, ne01_fd, ne02_fd,
                       ne11_fd, ne12_fd, item);
        });
}

// Dispatch over dst type for a given (src_t=float, idx_t)
template <typename idx_t>
static void set_rows_sycl(ggml_backend_sycl_context & ctx,
                           const ggml_tensor * src0, const ggml_tensor * src1,
                           ggml_tensor * dst) {
    const float * src0_d = (const float *) src0->data;
    const idx_t * src1_d = (const idx_t *) src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    sycl::queue & stream = *(ctx.stream());

    if (dst->type == GGML_TYPE_F32) {
        set_rows_sycl(
            src0_d, src1_d, (float *) dst->data,
            ne00, ne01, ne02, ne03,
            ne11, ne12,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream);
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_sycl(
            src0_d, src1_d, (sycl::half *) dst->data,
            ne00, ne01, ne02, ne03,
            ne11, ne12,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream);
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_sycl(
            src0_d, src1_d, (sycl_bf16 *) dst->data,
            ne00, ne01, ne02, ne03,
            ne11, ne12,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream);
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_sycl_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0 *) dst->data,
            ne00, ne01, ne02, ne03,
            ne11, ne12,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream);
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_sycl_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1 *) dst->data,
            ne00, ne01, ne02, ne03,
            ne11, ne12,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream);
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_sycl_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0 *) dst->data,
            ne00, ne01, ne02, ne03,
            ne11, ne12,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream);
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_sycl_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1 *) dst->data,
            ne00, ne01, ne02, ne03,
            ne11, ne12,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream);
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_sycl_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0 *) dst->data,
            ne00, ne01, ne02, ne03,
            ne11, ne12,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream);
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_sycl_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl *) dst->data,
            ne00, ne01, ne02, ne03,
            ne11, ne12,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream);
    } else {
        GGML_ABORT("set_rows: unsupported dst type %s", ggml_type_name(dst->type));
    }
}

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_sycl<int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_sycl<int32_t>(ctx, src0, src1, dst);
    }
}
