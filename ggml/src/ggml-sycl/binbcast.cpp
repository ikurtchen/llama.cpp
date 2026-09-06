// SYCL broadcasted binary ops. Mirrors ggml-cuda/binbcast.cu's logical-index broadcast rules,
// but is hand-written for SYCL with a flat nd_range kernel over dst elements.
#include "binbcast.hpp"

#include <sycl/sycl.hpp>

namespace {

struct op_repeat {
    float operator()(float a, float b) const {
        (void) a;
        return b;
    }
};

struct op_add {
    float operator()(float a, float b) const { return a + b; }
};

struct op_sub {
    float operator()(float a, float b) const { return a - b; }
};

struct op_mul {
    float operator()(float a, float b) const { return a * b; }
};

struct op_div {
    float operator()(float a, float b) const { return a / b; }
};

template <typename src0_t, typename src1_t, typename dst_t, typename op_t>
static void launch_binbcast(
        sycl::queue & q,
        const src0_t * src0,
        const src1_t * src1,
        dst_t * dst,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
        size_t nb0, size_t nb1, size_t nb2, size_t nb3,
        size_t nb00, size_t nb01, size_t nb02, size_t nb03,
        size_t nb10, size_t nb11, size_t nb12, size_t nb13) {
    const int64_t ne = ne0 * ne1 * ne2 * ne3;
    if (ne == 0) {
        return;
    }

    const int64_t block = 256;
    const int64_t nblk  = (ne + block - 1) / block;

    const char * src0_bytes = reinterpret_cast<const char *>(src0);
    const char * src1_bytes = reinterpret_cast<const char *>(src1);
    char * dst_bytes = reinterpret_cast<char *>(dst);

    q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (nblk * block)), sycl::range<1>((size_t) block)),
        [=](sycl::nd_item<1> item) {
            const int64_t i = (int64_t) item.get_global_id(0);
            if (i >= ne) {
                return;
            }

            const int64_t ne01_dst  = ne0 * ne1;
            const int64_t ne012_dst = ne01_dst * ne2;

            int64_t rem = i;
            const int64_t i3 = rem / ne012_dst;
            rem -= i3 * ne012_dst;
            const int64_t i2 = rem / ne01_dst;
            rem -= i2 * ne01_dst;
            const int64_t i1 = rem / ne0;
            const int64_t i0 = rem - i1 * ne0;

            const size_t dst_off =
                (size_t) i0 * nb0 +
                (size_t) i1 * nb1 +
                (size_t) i2 * nb2 +
                (size_t) i3 * nb3;

            const size_t src1_off =
                (size_t) (i0 % ne10) * nb10 +
                (size_t) (i1 % ne11) * nb11 +
                (size_t) (i2 % ne12) * nb12 +
                (size_t) (i3 % ne13) * nb13;

            float result = 0.0f;
            if (src0_bytes != nullptr) {
                const size_t src0_off =
                    (size_t) (i0 % ne00) * nb00 +
                    (size_t) (i1 % ne01) * nb01 +
                    (size_t) (i2 % ne02) * nb02 +
                    (size_t) (i3 % ne03) * nb03;
                result = (float) *reinterpret_cast<const src0_t *>(src0_bytes + src0_off);
            }

            const float rhs = (float) *reinterpret_cast<const src1_t *>(src1_bytes + src1_off);
            *reinterpret_cast<dst_t *>(dst_bytes + dst_off) = (dst_t) op_t{}(result, rhs);
        });
}

static bool supported_binbcast_types(ggml_type src0_type, ggml_type src1_type, ggml_type dst_type) {
    return
        (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) ||
        (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) ||
        (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) ||
        (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32);
}

template <typename op_t>
static void dispatch_binbcast(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_can_repeat(src1, src0));

    GGML_TENSOR_BINARY_OP_LOCALS

    sycl::queue & q = ctx.stream();

#define BINBCAST_DISPATCH(SRC0_T, SRC1_T, DST_T)                                                                  \
    launch_binbcast<SRC0_T, SRC1_T, DST_T, op_t>(q, (const SRC0_T *) src0->data, (const SRC1_T *) src1->data,    \
        (DST_T *) dst->data,                                                                                       \
        ne0, ne1, ne2, ne3,                                                                                        \
        ne00, ne01, ne02, ne03,                                                                                    \
        ne10, ne11, ne12, ne13,                                                                                    \
        nb0, nb1, nb2, nb3,                                                                                        \
        nb00, nb01, nb02, nb03,                                                                                    \
        nb10, nb11, nb12, nb13)

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        BINBCAST_DISPATCH(float, float, float);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        BINBCAST_DISPATCH(sycl::half, sycl::half, sycl::half);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        BINBCAST_DISPATCH(sycl::half, float, sycl::half);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        BINBCAST_DISPATCH(sycl::half, float, float);
    } else {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }

#undef BINBCAST_DISPATCH
}

static void dispatch_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    sycl::queue & q = ctx.stream();

#define REPEAT_DISPATCH(T)                                                                                         \
    launch_binbcast<T, T, T, op_repeat>(q, nullptr, (const T *) src0->data, (T *) dst->data,                     \
        ne0, ne1, ne2, ne3,                                                                                        \
        1, 1, 1, 1,                                                                                                \
        ne00, ne01, ne02, ne03,                                                                                    \
        nb0, nb1, nb2, nb3,                                                                                        \
        0, 0, 0, 0,                                                                                                \
        nb00, nb01, nb02, nb03)

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        REPEAT_DISPATCH(float);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        REPEAT_DISPATCH(sycl::half);
    } else {
        GGML_ABORT("%s: unsupported types for REPEAT: dst: %s, src0: %s", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
    }

#undef REPEAT_DISPATCH
}

static void dispatch_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_can_repeat(dst, src0));
    GGML_ASSERT(src0->ne[2] * src0->ne[3] <= (1 << 15));

    GGML_TENSOR_UNARY_OP_LOCALS

    const float * src = (const float *) src0->data;
    float * out = (float *) dst->data;

    const int64_t ne = ne0 * ne1 * ne2 * ne3;
    if (ne == 0) {
        return;
    }

    sycl::queue & q = ctx.stream();
    const int64_t block = 256;
    const int64_t nblk  = (ne + block - 1) / block;

    q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (nblk * block)), sycl::range<1>((size_t) block)),
        [=](sycl::nd_item<1> item) {
            const int64_t i = (int64_t) item.get_global_id(0);
            if (i >= ne) {
                return;
            }

            const int64_t ne01_dst  = ne0 * ne1;
            const int64_t ne012_dst = ne01_dst * ne2;

            int64_t rem = i;
            const int64_t i3 = rem / ne012_dst;
            rem -= i3 * ne012_dst;
            const int64_t i2 = rem / ne01_dst;
            rem -= i2 * ne01_dst;
            const int64_t i1 = rem / ne0;
            const int64_t i0 = rem - i1 * ne0;

            float sum = 0.0f;
            for (int64_t s3 = i3; s3 < ne03; s3 += ne3) {
                for (int64_t s2 = i2; s2 < ne02; s2 += ne2) {
                    for (int64_t s1 = i1; s1 < ne01; s1 += ne1) {
                        for (int64_t s0 = i0; s0 < ne00; s0 += ne0) {
                            const size_t src_off =
                                (size_t) s0 * nb00 +
                                (size_t) s1 * nb01 +
                                (size_t) s2 * nb02 +
                                (size_t) s3 * nb03;
                            sum += *(const float *) ((const char *) src + src_off);
                        }
                    }
                }
            }

            out[i] = sum;
        });
}

} // namespace

void ggml_sycl_op_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    dispatch_binbcast<op_add>(ctx, dst);
}

void ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    dispatch_binbcast<op_sub>(ctx, dst);
}

void ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    dispatch_binbcast<op_mul>(ctx, dst);
}

void ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    dispatch_binbcast<op_div>(ctx, dst);
}

void ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    dispatch_repeat(ctx, dst);
}

void ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    dispatch_repeat_back(ctx, dst);
}

bool ggml_sycl_supports_binbcast(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];

    switch (op->op) {
        case GGML_OP_REPEAT:
            return src0 != nullptr && ggml_can_repeat(src0, op) &&
                ((src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ||
                 (src0->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16));
        case GGML_OP_REPEAT_BACK:
            return src0 != nullptr && src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_can_repeat(op, src0) && (src0->ne[2] * src0->ne[3]) <= (1 << 15);
        case GGML_OP_ADD1:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV: {
            const ggml_tensor * src1 = op->src[1];
            return src0 != nullptr && src1 != nullptr && ggml_are_same_shape(src0, op) && ggml_can_repeat(src1, src0) &&
                   supported_binbcast_types(src0->type, src1->type, op->type);
        }
        default:
            return false;
    }
}
