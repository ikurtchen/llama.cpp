#include "tri.hpp"

#include <sycl/sycl.hpp>
#include <type_traits>

namespace {
template <typename T>
inline float to_float(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, sycl::half>) {
        return (float) value;
    } else {
        return GGML_BF16_TO_FP32(value);
    }
}

template <typename T>
inline T from_float(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, sycl::half>) {
        return (sycl::half) value;
    } else {
        return GGML_FP32_TO_BF16(value);
    }
}

template <typename T, bool prefix_keep, int add_to_split>
void tri_kernel(sycl::queue & q, const ggml_tensor * src0, ggml_tensor * dst) {
    const int64_t total = ggml_nelements(dst);
    const int64_t block = 256;
    const int64_t nblk  = (total + block - 1) / block;
    const char * src = (const char *) src0->data;
    char * out       = (char *) dst->data;
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];
    const size_t nb0  = dst->nb[0];
    const size_t nb1  = dst->nb[1];
    const size_t nb2  = dst->nb[2];
    const size_t nb3  = dst->nb[3];

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int64_t idx = item.get_global_id(0);
        if (idx >= total) {
            return;
        }

        int64_t rem = idx;
        const int64_t i0 = rem % ne0;
        rem /= ne0;
        const int64_t i1 = rem % ne1;
        rem /= ne1;
        const int64_t i2 = rem % ne2;
        const int64_t i3 = rem / ne2;

        const int64_t split_point = i1 + add_to_split;
        const bool keep = prefix_keep ? (i0 < split_point) : (i0 >= split_point);
        const size_t src_off = i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00;
        const size_t dst_off = i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0;

        const T value = keep ? *(const T *) (src + src_off) : from_float<T>(0.0f);
        *(T *) (out + dst_off) = value;
    });
}

template <typename T>
void tri_dispatch(sycl::queue & q, const ggml_tensor * src0, ggml_tensor * dst, ggml_tri_type type) {
    const int add_to_split = (type == GGML_TRI_TYPE_LOWER_DIAG || type == GGML_TRI_TYPE_UPPER) ? 1 : 0;
    const bool prefix_keep = type == GGML_TRI_TYPE_LOWER || type == GGML_TRI_TYPE_LOWER_DIAG;

    if (prefix_keep) {
        if (add_to_split == 0) {
            tri_kernel<T, true, 0>(q, src0, dst);
        } else {
            tri_kernel<T, true, 1>(q, src0, dst);
        }
    } else {
        if (add_to_split == 0) {
            tri_kernel<T, false, 0>(q, src0, dst);
        } else {
            tri_kernel<T, false, 1>(q, src0, dst);
        }
    }
}
}

void ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tri_type type = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(src0->type == dst->type);

    sycl::queue & q = ctx.stream();
    switch (src0->type) {
        case GGML_TYPE_F32:
            tri_dispatch<float>(q, src0, dst, type);
            break;
        case GGML_TYPE_F16:
            tri_dispatch<sycl::half>(q, src0, dst, type);
            break;
        case GGML_TYPE_BF16:
            tri_dispatch<ggml_bf16_t>(q, src0, dst, type);
            break;
        default:
            GGML_ABORT("%s: unsupported type %s", __func__, ggml_type_name(src0->type));
    }
}

bool ggml_sycl_supports_tri(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return src0->type == op->type &&
           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16);
}
