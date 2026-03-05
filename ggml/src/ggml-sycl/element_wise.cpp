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
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

static float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;
    return 0.5f * x * (1.0f + erff(x * SQRT_2_INV));
}

static float op_gelu_quick(float x) {
    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

static float op_silu(float x) { return x / (1.0f + expf(-x)); }

static float op_tanh(float x) { return tanhf(x); }

static float op_relu(float x) { return fmaxf(x, 0); }

static float op_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static float op_hardsigmoid(float x) { return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f)); }

static float op_hardswish(float x) { return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f)); }

static float op_exp(float x) { return expf(x); }

static float op_sqr(float x) { return x * x; }

static float op_sqrt(float x) { return sqrtf(x); }

static float op_sin(float x) { return sinf(x); }

static float op_cos(float x) { return cosf(x); }

static float op_log(float x) { return logf(x); }

static float op_expm1(float x) { return expm1f(x); }

static float op_softplus(float x) { return (x > 20.0f) ? x : logf(1.0f + expf(x)); }

static float op_elu(float x) { return (x > 0.f) ? x : expm1f(x); }

static float op_floor(float x) { return floorf(x); }

static float op_ceil(float x) { return ceilf(x); }

static float op_round(float x) { return roundf(x); }

static float op_trunc(float x) { return truncf(x); }

// ============================================================================
// Generic unary kernel dispatch templates
// Mirrors the CUDA unary_op_kernel / unary_cuda / ggml_cuda_op_unary pattern.
// ============================================================================

// Per-element unary kernel (float)
template <float (*op)(float), typename T>
static void unary_op_kernel(const T * x, T * dst, const int k,
                            const sycl::nd_item<1> & item) {
    const int i = item.get_global_id(0);
    if (i >= k) {
        return;
    }
    dst[i] = (T)op((float)x[i]);
}

// SYCL launch wrapper
template <float (*op)(float), typename T>
static void unary_sycl(const T * x, T * dst, const int k, sycl::queue & stream) {
    const int num_blocks = (k + SYCL_NEG_BLOCK_SIZE - 1) / SYCL_NEG_BLOCK_SIZE;
    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_NEG_BLOCK_SIZE, SYCL_NEG_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            unary_op_kernel<op>(x, dst, k, item);
        });
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
    float s = 1.0f / (1.0f + expf(-x));
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
    float y_neg = (expm1f(min_v_eps) - xi) * alpha_n + beta * xi;
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
