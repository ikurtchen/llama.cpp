#include "snake.hpp"

#include <sycl/sycl.hpp>
#include <array>
#include <type_traits>

namespace {
static constexpr std::array<ggml_op, 5> snake_pattern = {
    GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD,
};

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

template <typename T>
void snake_kernel(sycl::queue & q, const T * x, const float * a, const float * inv_b, T * dst, int total, int t_len) {
    const int block = 256;
    const int grid = (total + block - 1) / block;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int idx = (int) item.get_global_id(0);
        if (idx >= total) {
            return;
        }

        const int c = idx / t_len;
        const float xi = to_float(x[idx]);
        const float s = sycl::sin(a[c] * xi);
        dst[idx] = from_float<T>(xi + s * s * inv_b[c]);
    });
}
}

bool ggml_sycl_can_fuse_snake(const ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_can_fuse(cgraph, node_idx, snake_pattern.begin(), (int) snake_pattern.size())) {
        return false;
    }

    const ggml_tensor * mul0     = cgraph->nodes[node_idx + 0];
    const ggml_tensor * sin_node = cgraph->nodes[node_idx + 1];
    const ggml_tensor * sqr      = cgraph->nodes[node_idx + 2];
    const ggml_tensor * mul1     = cgraph->nodes[node_idx + 3];
    const ggml_tensor * add      = cgraph->nodes[node_idx + 4];

    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = x == mul0->src[0] ? mul0->src[1] : mul0->src[0];
    const ggml_tensor * inv_b = mul1->src[0] == sqr ? mul1->src[1] : mul1->src[0];
    const ggml_tensor * x_in_add = add->src[0] == mul1 ? add->src[1] : add->src[0];

    if (x_in_add != x) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_BF16) {
        return false;
    }
    if (a->type != GGML_TYPE_F32 || inv_b->type != GGML_TYPE_F32) {
        return false;
    }
    if (mul0->type != x->type || sin_node->type != x->type || sqr->type != x->type || mul1->type != x->type || add->type != x->type) {
        return false;
    }
    if (!ggml_are_same_shape(a, inv_b) || a->ne[0] != 1 || a->ne[1] != x->ne[1]) {
        return false;
    }
    if (x->ne[2] != 1 || x->ne[3] != 1 || add->ne[2] != 1 || add->ne[3] != 1 ||
        a->ne[2] != 1 || a->ne[3] != 1 || inv_b->ne[2] != 1 || inv_b->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(add) || !ggml_is_contiguous(a) || !ggml_is_contiguous(inv_b)) {
        return false;
    }
    return true;
}

void ggml_sycl_op_snake_fused(ggml_backend_sycl_context & ctx,
                              const ggml_tensor * x,
                              const ggml_tensor * a,
                              const ggml_tensor * inv_b,
                              ggml_tensor * dst) {
    const int t_len = (int) x->ne[0];
    const int c_len = (int) x->ne[1];
    const int total = t_len * c_len;

    const float * a_d = (const float *) a->data;
    const float * inv_b_d = (const float *) inv_b->data;
    sycl::queue & q = ctx.stream();

    switch (x->type) {
        case GGML_TYPE_F32:
            snake_kernel(q, (const float *) x->data, a_d, inv_b_d, (float *) dst->data, total, t_len);
            break;
        case GGML_TYPE_F16:
            snake_kernel(q, (const sycl::half *) x->data, a_d, inv_b_d, (sycl::half *) dst->data, total, t_len);
            break;
        case GGML_TYPE_BF16:
            snake_kernel(q, (const ggml_bf16_t *) x->data, a_d, inv_b_d, (ggml_bf16_t *) dst->data, total, t_len);
            break;
        default:
            GGML_ABORT("%s: unsupported type %s", __func__, ggml_type_name(x->type));
    }
}
