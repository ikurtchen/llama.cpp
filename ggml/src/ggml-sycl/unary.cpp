// SYCL unary + GLU (gated) elementwise ops. Mirrors ggml-cuda/unary.cu's per-element math (silu,
// gelu, relu, swiglu, ...) but written directly against SYCL, not translated from the CUDA source.
#include "unary.hpp"

#include <sycl/sycl.hpp>

namespace {

inline float op_abs(float x)          { return sycl::fabs(x); }
inline float op_sgn(float x)          { return x > 0.f ? 1.f : (x < 0.f ? -1.f : 0.f); }
inline float op_neg(float x)          { return -x; }
inline float op_step(float x)         { return x > 0.0f; }
inline float op_silu(float x)         { return x / (1.0f + sycl::native::exp(-x)); }
inline float op_gelu(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + sycl::tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}
inline float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;
    return 0.5f * x * (1.0f + sycl::erf(x * SQRT_2_INV));
}
inline float op_gelu_quick(float x) {
    const float GELU_QUICK_COEF = -1.702f;
    return x * (1.0f / (1.0f + sycl::native::exp(GELU_QUICK_COEF * x)));
}
inline float op_tanh(float x)         { return sycl::tanh(x); }
inline float op_relu(float x)         { return sycl::fmax(x, 0.0f); }
inline float op_sigmoid(float x)      { return 1.0f / (1.0f + sycl::native::exp(-x)); }
inline float op_hardsigmoid(float x)  { return sycl::fmin(1.0f, sycl::fmax(0.0f, (x + 3.0f) / 6.0f)); }
inline float op_hardswish(float x)    { return x * sycl::fmin(1.0f, sycl::fmax(0.0f, (x + 3.0f) / 6.0f)); }
inline float op_exp(float x)          { return sycl::native::exp(x); }
inline float op_sqr(float x)          { return x * x; }
inline float op_sqrt(float x)         { return sycl::sqrt(x); }
inline float op_sin(float x)          { return sycl::sin(x); }
inline float op_cos(float x)          { return sycl::cos(x); }
inline float op_log(float x)          { return sycl::log(x); }
inline float op_expm1(float x)        { return sycl::expm1(x); }
inline float op_softplus(float x)     { return (x > 20.0f) ? x : sycl::log(1.0f + sycl::native::exp(x)); }
inline float op_elu(float x)          { return (x > 0.f) ? x : sycl::expm1(x); }
inline float op_floor(float x)        { return sycl::floor(x); }
inline float op_ceil(float x)         { return sycl::ceil(x); }
inline float op_round(float x)        { return sycl::round(x); }
inline float op_trunc(float x)        { return sycl::trunc(x); }

template <typename T, float (*op)(float)>
void launch_unary(sycl::queue & q, const T * x, T * dst, int64_t k) {
    const int64_t block = 256;
    const int64_t nblk  = (k + block - 1) / block;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int64_t i = item.get_global_id(0);
        if (i >= k) {
            return;
        }
        dst[i] = (T) op((float) x[i]);
    });
}

template <float (*op)(float)>
void dispatch_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    sycl::queue & q  = ctx.stream();
    const int64_t ne = ggml_nelements(src0);
    if (src0->type == GGML_TYPE_F16) {
        launch_unary<sycl::half, op>(q, (const sycl::half *) src0->data, (sycl::half *) dst->data, ne);
    } else {
        launch_unary<float, op>(q, (const float *) src0->data, (float *) dst->data, ne);
    }
}

// Gated op: dst[i] = op(x[j0]) * g[j1], where x/g may be two halves of one tensor (src1==null) or
// two separate tensors. Mirrors ggml-cuda/unary.cu's unary_gated_op_kernel indexing scheme.
template <typename T, float (*op)(float)>
void launch_gated(sycl::queue & q, const T * x, const T * g, T * dst, int64_t k, int64_t n, int64_t o0, int64_t o1) {
    const int64_t block = 256;
    const int64_t nblk  = (k + block - 1) / block;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int64_t i = item.get_global_id(0);
        if (i >= k) {
            return;
        }
        const int64_t j0 = (i / n) * o0 + (i % n);
        const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);
        dst[i] = (T) (op((float) x[j0]) * (float) g[j1]);
    });
}

template <typename T>
void unpack_gated_inputs(const ggml_tensor * dst, int32_t swapped, int64_t nc,
                          const T ** src0_p, const T ** src1_p) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    *src0_p = (const T *) src0->data;
    *src1_p = src1 ? (const T *) src1->data : (const T *) src0->data;
    if (!src1) {
        *src0_p += swapped ? nc : 0;
        *src1_p += swapped ? 0 : nc;
    }
}

template <float (*op)(float)>
void dispatch_gated(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    const int64_t nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    sycl::queue & q = ctx.stream();
    if (src0->type == GGML_TYPE_F16) {
        const sycl::half * p0; const sycl::half * p1;
        unpack_gated_inputs<sycl::half>(dst, swapped, nc, &p0, &p1);
        launch_gated<sycl::half, op>(q, p0, p1, (sycl::half *) dst->data, ggml_nelements(dst), nc,
                                      src0_o / sizeof(sycl::half), src1_o / sizeof(sycl::half));
    } else {
        const float * p0; const float * p1;
        unpack_gated_inputs<float>(dst, swapped, nc, &p0, &p1);
        launch_gated<float, op>(q, p0, p1, (float *) dst->data, ggml_nelements(dst), nc,
                                 src0_o / sizeof(float), src1_o / sizeof(float));
    }
}

inline float op_swiglu_oai_single(float x, float g, float alpha, float limit) {
    x = sycl::fmin(x, limit);
    g = sycl::fmax(sycl::fmin(g, limit), -limit);
    float out_glu = x / (1.0f + sycl::native::exp(-x * alpha));
    return out_glu * (1.0f + g);
}

void dispatch_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    const int64_t nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float   alpha   = ggml_get_op_params_f32(dst, 2);
    const float   limit   = ggml_get_op_params_f32(dst, 3);

    const float * p0; const float * p1;
    unpack_gated_inputs<float>(dst, swapped, nc, &p0, &p1);

    sycl::queue & q  = ctx.stream();
    float * dst_d    = (float *) dst->data;
    const int64_t k  = ggml_nelements(dst);
    const int64_t o0 = src0_o / sizeof(float);
    const int64_t o1 = src1_o / sizeof(float);
    const int64_t block = 256;
    const int64_t nblk   = (k + block - 1) / block;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int64_t i = item.get_global_id(0);
        if (i >= k) {
            return;
        }
        const int64_t j0 = (i / nc) * o0 + (i % nc);
        const int64_t j1 = o0 == o1 ? j0 : (i / nc) * o1 + (i % nc);
        dst_d[i] = op_swiglu_oai_single(p0[j0], p1[j1], alpha, limit);
    });
}

} // namespace

void ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == src0->type);

    switch (ggml_get_unary_op(dst)) {
        case GGML_UNARY_OP_ABS:          dispatch_unary<op_abs>(ctx, dst); break;
        case GGML_UNARY_OP_SGN:          dispatch_unary<op_sgn>(ctx, dst); break;
        case GGML_UNARY_OP_NEG:          dispatch_unary<op_neg>(ctx, dst); break;
        case GGML_UNARY_OP_STEP:         dispatch_unary<op_step>(ctx, dst); break;
        case GGML_UNARY_OP_GELU:         dispatch_unary<op_gelu>(ctx, dst); break;
        case GGML_UNARY_OP_SILU:         dispatch_unary<op_silu>(ctx, dst); break;
        case GGML_UNARY_OP_GELU_ERF:     dispatch_unary<op_gelu_erf>(ctx, dst); break;
        case GGML_UNARY_OP_GELU_QUICK:   dispatch_unary<op_gelu_quick>(ctx, dst); break;
        case GGML_UNARY_OP_TANH:         dispatch_unary<op_tanh>(ctx, dst); break;
        case GGML_UNARY_OP_RELU:         dispatch_unary<op_relu>(ctx, dst); break;
        case GGML_UNARY_OP_SIGMOID:      dispatch_unary<op_sigmoid>(ctx, dst); break;
        case GGML_UNARY_OP_HARDSIGMOID:  dispatch_unary<op_hardsigmoid>(ctx, dst); break;
        case GGML_UNARY_OP_HARDSWISH:    dispatch_unary<op_hardswish>(ctx, dst); break;
        case GGML_UNARY_OP_EXP:          dispatch_unary<op_exp>(ctx, dst); break;
        case GGML_UNARY_OP_ELU:          dispatch_unary<op_elu>(ctx, dst); break;
        case GGML_UNARY_OP_FLOOR:        dispatch_unary<op_floor>(ctx, dst); break;
        case GGML_UNARY_OP_CEIL:         dispatch_unary<op_ceil>(ctx, dst); break;
        case GGML_UNARY_OP_ROUND:        dispatch_unary<op_round>(ctx, dst); break;
        case GGML_UNARY_OP_TRUNC:        dispatch_unary<op_trunc>(ctx, dst); break;
        case GGML_UNARY_OP_EXPM1:        dispatch_unary<op_expm1>(ctx, dst); break;
        case GGML_UNARY_OP_SOFTPLUS:     dispatch_unary<op_softplus>(ctx, dst); break;
        default:
            GGML_ABORT("%s: unsupported unary op", __func__);
    }
}

bool ggml_sycl_supports_unary(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    auto ok_type = [](ggml_type t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16; };
    if (!ggml_is_contiguous(src0) || !ok_type(src0->type) || op->type != src0->type) {
        return false;
    }
    switch (ggml_get_unary_op(op)) {
        case GGML_UNARY_OP_ABS: case GGML_UNARY_OP_SGN: case GGML_UNARY_OP_NEG: case GGML_UNARY_OP_STEP:
        case GGML_UNARY_OP_GELU: case GGML_UNARY_OP_SILU: case GGML_UNARY_OP_GELU_ERF:
        case GGML_UNARY_OP_GELU_QUICK: case GGML_UNARY_OP_TANH: case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_SIGMOID: case GGML_UNARY_OP_HARDSIGMOID: case GGML_UNARY_OP_HARDSWISH:
        case GGML_UNARY_OP_EXP: case GGML_UNARY_OP_ELU: case GGML_UNARY_OP_FLOOR: case GGML_UNARY_OP_CEIL:
        case GGML_UNARY_OP_ROUND: case GGML_UNARY_OP_TRUNC: case GGML_UNARY_OP_EXPM1: case GGML_UNARY_OP_SOFTPLUS:
            return true;
        default:
            return false;
    }
}

void ggml_sycl_op_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);

    switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_REGLU:      dispatch_gated<op_relu>(ctx, dst); break;
        case GGML_GLU_OP_GEGLU:      dispatch_gated<op_gelu>(ctx, dst); break;
        case GGML_GLU_OP_SWIGLU:     dispatch_gated<op_silu>(ctx, dst); break;
        case GGML_GLU_OP_GEGLU_ERF:  dispatch_gated<op_gelu_erf>(ctx, dst); break;
        case GGML_GLU_OP_GEGLU_QUICK: dispatch_gated<op_gelu_quick>(ctx, dst); break;
        case GGML_GLU_OP_SWIGLU_OAI: dispatch_swiglu_oai(ctx, dst); break;
        default:
            GGML_ABORT("%s: unsupported glu op", __func__);
    }
}

bool ggml_sycl_supports_glu(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    auto ok_type = [](ggml_type t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16; };
    if (!ok_type(src0->type)) {
        return false;
    }
    switch (ggml_get_glu_op(op)) {
        case GGML_GLU_OP_REGLU: case GGML_GLU_OP_GEGLU: case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_GEGLU_ERF: case GGML_GLU_OP_GEGLU_QUICK:
            return true;
        case GGML_GLU_OP_SWIGLU_OAI:
            return src0->type == GGML_TYPE_F32; // swiglu_oai is F32-only upstream too
        default:
            return false;
    }
}

void ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst)  { dispatch_unary<op_sqr>(ctx, dst); }
void ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) { dispatch_unary<op_sqrt>(ctx, dst); }
void ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst)  { dispatch_unary<op_sin>(ctx, dst); }
void ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst)  { dispatch_unary<op_cos>(ctx, dst); }
void ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst)  { dispatch_unary<op_log>(ctx, dst); }

bool ggml_sycl_supports_unary_f32f16(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    auto ok_type = [](ggml_type t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16; };
    return ok_type(src0->type) && op->type == src0->type;
}
