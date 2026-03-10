#include "element_wise.hpp"
#include "ggml-impl.h"

#include <cmath>
#include <cstring>

// ============================================================================
// Device-side activation functions
// Math copied from ggml-cuda/unary.cu to ensure numerical equivalence.
// ============================================================================

static float op_abs(float x) { return fabsf(x); }

static float op_sgn(float x) { return (x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f))); }

static float op_neg(float x) { return -x; }

static float op_step(float x) { return x > 0.0f; }

static float op_gelu(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    float arg = SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x);
    // tanh(arg) = 1 - 2/(exp(2*arg) + 1), using native exp
    float e2a = sycl::native::exp(2.0f * arg);
    float t   = 1.0f - 2.0f / (e2a + 1.0f);
    return 0.5f * x * (1.0f + t);
}

static float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;
    return 0.5f * x * (1.0f + erff(x * SQRT_2_INV));
}

static float op_gelu_quick(float x) {
    return x * (1.0f / (1.0f + sycl::native::exp(GELU_QUICK_COEF * x)));
}

static float op_silu(float x) { return x / (1.0f + sycl::native::exp(-x)); }

static float op_tanh(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;
    float e2x = sycl::native::exp(2.0f * x);
    return 1.0f - 2.0f / (e2x + 1.0f);
}

static float op_relu(float x) { return fmaxf(x, 0); }

static float op_sigmoid(float x) { return 1.0f / (1.0f + sycl::native::exp(-x)); }

static float op_hardsigmoid(float x) { return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f)); }

static float op_hardswish(float x) { return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f)); }

static float op_exp(float x) { return sycl::native::exp(x); }

static float op_sqr(float x) { return x * x; }

static float op_sqrt(float x) { return sycl::native::sqrt(x); }

static float op_sin(float x) { return sycl::native::sin(x); }

static float op_cos(float x) { return sycl::native::cos(x); }

static float op_log(float x) { return sycl::native::log(x); }

static float op_expm1(float x) { return sycl::native::exp(x) - 1.0f; }

static float op_softplus(float x) { return (x > 20.0f) ? x : sycl::native::log(1.0f + sycl::native::exp(x)); }

static float op_elu(float x) { return (x > 0.f) ? x : sycl::native::exp(x) - 1.0f; }

static float op_floor(float x) { return floorf(x); }

static float op_ceil(float x) { return ceilf(x); }

static float op_round(float x) { return roundf(x); }

static float op_trunc(float x) { return truncf(x); }

// ============================================================================
// Generic unary kernel dispatch templates
// Mirrors the CUDA unary_op_kernel / unary_cuda / ggml_cuda_op_unary pattern.
// Vec4 vectorization: each work-item processes 4 elements for better memory
// throughput on Xe2 GPUs.
// ============================================================================

// Vec4 unary kernel — each work-item loads/stores 4 elements
template <float (*op)(float)>
static void unary_op_kernel_vec4_f32(const float * x, float * dst, const int k4,
                                      const sycl::nd_item<1> & item) {
    const int i = item.get_global_id(0);
    if (i >= k4) {
        return;
    }
    const int base = i * 4;
    const sycl::float4 in = *reinterpret_cast<const sycl::float4 *>(x + base);
    sycl::float4 out;
    out.s0() = op(in.s0());
    out.s1() = op(in.s1());
    out.s2() = op(in.s2());
    out.s3() = op(in.s3());
    *reinterpret_cast<sycl::float4 *>(dst + base) = out;
}

template <float (*op)(float)>
static void unary_op_kernel_vec4_f16(const sycl::half * x, sycl::half * dst, const int k4,
                                      const sycl::nd_item<1> & item) {
    const int i = item.get_global_id(0);
    if (i >= k4) {
        return;
    }
    const int base = i * 4;
    // Load 4 halfs (8 bytes) as a single 64-bit load via uint64_t
    const uint64_t bits = *reinterpret_cast<const uint64_t *>(x + base);
    const sycl::half * hp = reinterpret_cast<const sycl::half *>(&bits);
    sycl::half out[4];
    out[0] = (sycl::half)op((float)hp[0]);
    out[1] = (sycl::half)op((float)hp[1]);
    out[2] = (sycl::half)op((float)hp[2]);
    out[3] = (sycl::half)op((float)hp[3]);
    *reinterpret_cast<uint64_t *>(dst + base) = *reinterpret_cast<const uint64_t *>(out);
}

// Scalar tail kernel for remaining elements (k % 4 != 0)
template <float (*op)(float), typename T>
static void unary_op_kernel_scalar(const T * x, T * dst, const int k,
                                    const int offset,
                                    const sycl::nd_item<1> & item) {
    const int i = offset + item.get_global_id(0);
    if (i >= k) {
        return;
    }
    dst[i] = (T)op((float)x[i]);
}

// SYCL launch wrapper — vec4 path with scalar tail
template <float (*op)(float)>
static void unary_sycl(const float * x, float * dst, const int k, sycl::queue & stream) {
    const int k4 = k / 4;
    if (k4 > 0) {
        const int num_blocks = (k4 + SYCL_NEG_BLOCK_SIZE - 1) / SYCL_NEG_BLOCK_SIZE;
        stream.parallel_for(
            sycl::nd_range<1>(num_blocks * SYCL_NEG_BLOCK_SIZE, SYCL_NEG_BLOCK_SIZE),
            [=](sycl::nd_item<1> item) {
                unary_op_kernel_vec4_f32<op>(x, dst, k4, item);
            });
    }
    const int tail = k - k4 * 4;
    if (tail > 0) {
        const int tail_blocks = (tail + SYCL_NEG_BLOCK_SIZE - 1) / SYCL_NEG_BLOCK_SIZE;
        stream.parallel_for(
            sycl::nd_range<1>(tail_blocks * SYCL_NEG_BLOCK_SIZE, SYCL_NEG_BLOCK_SIZE),
            [=](sycl::nd_item<1> item) {
                unary_op_kernel_scalar<op>(x, dst, k, k4 * 4, item);
            });
    }
}

template <float (*op)(float)>
static void unary_sycl(const sycl::half * x, sycl::half * dst, const int k, sycl::queue & stream) {
    const int k4 = k / 4;
    if (k4 > 0) {
        const int num_blocks = (k4 + SYCL_NEG_BLOCK_SIZE - 1) / SYCL_NEG_BLOCK_SIZE;
        stream.parallel_for(
            sycl::nd_range<1>(num_blocks * SYCL_NEG_BLOCK_SIZE, SYCL_NEG_BLOCK_SIZE),
            [=](sycl::nd_item<1> item) {
                unary_op_kernel_vec4_f16<op>(x, dst, k4, item);
            });
    }
    const int tail = k - k4 * 4;
    if (tail > 0) {
        const int tail_blocks = (tail + SYCL_NEG_BLOCK_SIZE - 1) / SYCL_NEG_BLOCK_SIZE;
        stream.parallel_for(
            sycl::nd_range<1>(tail_blocks * SYCL_NEG_BLOCK_SIZE, SYCL_NEG_BLOCK_SIZE),
            [=](sycl::nd_item<1> item) {
                unary_op_kernel_scalar<op>(x, dst, k, k4 * 4, item);
            });
    }
}

// Host dispatch: extract pointers from ggml_tensor and launch
template <float (*op)(float)>
static void ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    sycl::queue & stream = *(ctx.stream());

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        unary_sycl<op>((const sycl::half *)src0_d, (sycl::half *)dst_d,
                        ggml_nelements(src0), stream);
    } else {
        unary_sycl<op>((const float *)src0_d, (float *)dst_d,
                        ggml_nelements(src0), stream);
    }
}

// ============================================================================
// Generic gated unary kernel dispatch templates
// Mirrors the CUDA unary_gated_op_kernel / unary_gated_cuda /
// ggml_cuda_op_unary_gated pattern.
// ============================================================================

template <float (*op)(float), typename T>
static void unary_gated_op_kernel(const T * x, const T * g, T * dst,
                                   const int64_t k, const int64_t n,
                                   const int64_t o0, const int64_t o1,
                                   const sycl::nd_item<1> & item) {
    const int64_t i = item.get_global_id(0);
    if (i >= k) {
        return;
    }
    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    dst[i] = (T)(op((float)x[j0]) * (float)g[j1]);
}

template <float (*op)(float), typename T>
static void unary_gated_sycl(const T * x, const T * g, T * dst,
                               const int64_t k, const int64_t n,
                               const int64_t o0, const int64_t o1,
                               sycl::queue & stream) {
    const int64_t num_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_GLU_BLOCK_SIZE, SYCL_GLU_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            unary_gated_op_kernel<op>(x, g, dst, k, n, o0, o1, item);
        });
}

template <float (*op)(float)>
static void ggml_sycl_op_unary_gated(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    sycl::queue & stream = *(ctx.stream());

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ((const int32_t *) dst->op_params)[1];

    if (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_sycl<op>(src0_p, src1_p, (sycl::half *)dst_d,
                              ggml_nelements(dst), nc,
                              src0_o / sizeof(sycl::half),
                              src1_o / sizeof(sycl::half), stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_sycl<op>(src0_p, src1_p, (float *)dst_d,
                              ggml_nelements(dst), nc,
                              src0_o / sizeof(float),
                              src1_o / sizeof(float), stream);
    }
}

// ============================================================================
// Concrete ggml_sycl_* wrapper functions
// These match the declarations in element_wise.hpp.
// ============================================================================

void ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_abs>(ctx, dst);
}

void ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_sgn>(ctx, dst);
}

void ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_neg>(ctx, dst);
}

void ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_step>(ctx, dst);
}

void ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_gelu>(ctx, dst);
}

void ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_gelu_erf>(ctx, dst);
}

void ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_gelu_quick>(ctx, dst);
}

void ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_silu>(ctx, dst);
}

void ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_tanh>(ctx, dst);
}

void ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_relu>(ctx, dst);
}

void ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_sigmoid>(ctx, dst);
}

void ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_hardsigmoid>(ctx, dst);
}

void ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_hardswish>(ctx, dst);
}

void ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_exp>(ctx, dst);
}

void ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_sqr>(ctx, dst);
}

void ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_sqrt>(ctx, dst);
}

void ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_sin>(ctx, dst);
}

void ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_cos>(ctx, dst);
}

void ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_log>(ctx, dst);
}

void ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_softplus>(ctx, dst);
}

void ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_elu>(ctx, dst);
}

void ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_floor>(ctx, dst);
}

void ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_ceil>(ctx, dst);
}

void ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_round>(ctx, dst);
}

void ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_trunc>(ctx, dst);
}

// Gated activation wrappers

void ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary_gated<op_gelu>(ctx, dst);
}

void ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary_gated<op_relu>(ctx, dst);
}

void ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary_gated<op_silu>(ctx, dst);
}

void ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary_gated<op_gelu_erf>(ctx, dst);
}

void ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary_gated<op_gelu_quick>(ctx, dst);
}

// ============================================================================
// swiglu_oai — special gated activation (not a simple templated gated op)
// Formula: x_clamped = min(x, limit)
//          g_clamped = clamp(g, -limit, limit)
//          out = x_clamped / (1 + exp(-x_clamped * alpha)) * (1 + g_clamped)
// Mirrors ggml_cuda_op_swiglu_oai_single / swiglu_oai_kernel from unary.cu.
// ============================================================================

static float swiglu_oai_single(float x, float g, float alpha, float limit) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + sycl::native::exp(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}

template <typename T>
static void swiglu_oai_op_kernel(const T * x, const T * g, T * dst,
                                  const int64_t k, const int64_t n,
                                  const int64_t o0, const int64_t o1,
                                  const float alpha, const float limit,
                                  const sycl::nd_item<1> & item) {
    const int64_t i = item.get_global_id(0);
    if (i >= k) {
        return;
    }

    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    float xi = (float)x[j0];
    float gi = (float)g[j1];

    dst[i] = (T)swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
static void swiglu_oai_sycl(const T * x, const T * g, T * dst,
                              const int64_t k, const int64_t n,
                              const int64_t o0, const int64_t o1,
                              const float alpha, const float limit,
                              sycl::queue & stream) {
    const int64_t num_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_GLU_BLOCK_SIZE, SYCL_GLU_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            swiglu_oai_op_kernel(x, g, dst, k, n, o0, o1, alpha, limit, item);
        });
}

void ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    sycl::queue & stream = *(ctx.stream());

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    // swiglu_oai only supports F32 (matching CUDA)
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    float * src0_p = (float *) src0_d;
    float * src1_p = (float *) src1_d;

    if (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (float *)dst_d,
                      ggml_nelements(dst), nc,
                      src0_o / sizeof(float),
                      src1_o / sizeof(float),
                      alpha, limit, stream);
}

// ============================================================================
// Non-trivial unary ops (require extra parameters or multiple inputs)
// ============================================================================

void ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_unary<op_expm1>(ctx, dst);
}

// leaky_relu: needs negative_slope from op_params

template <typename T>
static void leaky_relu_op_kernel(const T * x, T * dst, const int k, const float negative_slope,
                                  const sycl::nd_item<1> & item) {
    const int i = item.get_global_id(0);
    if (i >= k) return;
    float xi = (float)x[i];
    dst[i] = (T)(fmaxf(xi, 0.0f) + fminf(xi, 0.0f) * negative_slope);
}

template <typename T>
static void leaky_relu_sycl(const T * x, T * dst, const int k, const float negative_slope, sycl::queue & stream) {
    const int num_blocks = (k + SYCL_NEG_BLOCK_SIZE - 1) / SYCL_NEG_BLOCK_SIZE;
    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_NEG_BLOCK_SIZE, SYCL_NEG_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            leaky_relu_op_kernel(x, dst, k, negative_slope, item);
        });
}

void ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    sycl::queue & stream = *(ctx.stream());

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        leaky_relu_sycl((const sycl::half *)src0_d, (sycl::half *)dst_d, ggml_nelements(src0), negative_slope, stream);
    } else {
        leaky_relu_sycl((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), negative_slope, stream);
    }
}

// silu_back: needs two inputs (grad and x)

template <typename T>
static void silu_back_op_kernel(const T * grad, const T * xf, T * dst, const int k,
                                 const sycl::nd_item<1> & item) {
    const int i = item.get_global_id(0);
    if (i >= k) return;
    float g = (float)grad[i];
    float x = (float)xf[i];
    float s = 1.0f / (1.0f + sycl::native::exp(-x));
    dst[i] = (T)(g * s * (1.0f + x * (1.0f - s)));
}

template <typename T>
static void silu_back_sycl(const T * grad, const T * x, T * dst, const int k, sycl::queue & stream) {
    const int num_blocks = (k + SYCL_NEG_BLOCK_SIZE - 1) / SYCL_NEG_BLOCK_SIZE;
    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_NEG_BLOCK_SIZE, SYCL_NEG_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            silu_back_op_kernel(grad, x, dst, k, item);
        });
}

void ggml_sycl_silu_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // input from forward pass
    const ggml_tensor * src1 = dst->src[1]; // grads of forward pass output

    const void * src0_d = src0->data;
    const void * src1_d = src1->data;
    void * dst_d = dst->data;
    sycl::queue & stream = *(ctx.stream());

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        silu_back_sycl((const sycl::half *)src0_d, (const sycl::half *)src1_d, (sycl::half *)dst_d, ggml_nelements(src0), stream);
    } else {
        silu_back_sycl((const float *)src0_d, (const float *)src1_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

// xielu: needs parameters from op_params

template <typename T>
static void xielu_op_kernel(const T * x, T * dst, const int k,
                             float alpha_n, float alpha_p, float beta, float eps,
                             const sycl::nd_item<1> & item) {
    const int i = item.get_global_id(0);
    if (i >= k) return;
    float xi = (float)x[i];
    float gate_pos = (xi > 0.0f) ? 1.0f : 0.0f;
    float y_pos = alpha_p * xi * xi + beta * xi;
    float min_v_eps = fminf(xi, eps);
    float y_neg = (sycl::native::exp(min_v_eps) - 1.0f - xi) * alpha_n + beta * xi;
    float out = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    dst[i] = (T)out;
}

template <typename T>
static void xielu_sycl(const T * x, T * dst, const int k,
                        float alpha_n, float alpha_p, float beta, float eps,
                        sycl::queue & stream) {
    const int num_blocks = (k + SYCL_NEG_BLOCK_SIZE - 1) / SYCL_NEG_BLOCK_SIZE;
    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_NEG_BLOCK_SIZE, SYCL_NEG_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            xielu_op_kernel(x, dst, k, alpha_n, alpha_p, beta, eps, item);
        });
}

void ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    sycl::queue & stream = *(ctx.stream());

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta    = ggml_get_op_params_f32(dst, 3);
    const float eps     = ggml_get_op_params_f32(dst, 4);

    if (src0->type == GGML_TYPE_F16) {
        xielu_sycl((const sycl::half *)src0_d, (sycl::half *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    } else {
        xielu_sycl((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    }
}
