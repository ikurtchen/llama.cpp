#include "conv2d.hpp"

#include <sycl/sycl.hpp>
#include <type_traits>

namespace {
static inline int64_t max_i64(int64_t a, int64_t b) { return a > b ? a : b; }
static inline int64_t min_i64(int64_t a, int64_t b) { return a < b ? a : b; }

struct conv_params {
    int64_t iw;
    int64_t ih;
    int64_t ow;
    int64_t oh;
    int64_t kw;
    int64_t kh;
    int64_t st_x;
    int64_t st_y;
    int64_t pd_x;
    int64_t pd_y;
    int64_t dl_x;
    int64_t dl_y;
    int64_t ic;
    int64_t oc;
    int64_t batches;
    int64_t total;
};

struct kernel_bounds {
    int64_t y_min;
    int64_t y_max;
    int64_t x_min;
    int64_t x_max;
};

static inline kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, const conv_params & p) {
    kernel_bounds bounds;
    bounds.y_min = max_i64(0, (p.pd_y - out_y * p.st_y + p.dl_y - 1) / p.dl_y);
    bounds.y_max = min_i64(p.kh, (p.ih + p.pd_y - out_y * p.st_y + p.dl_y - 1) / p.dl_y);
    bounds.x_min = max_i64(0, (p.pd_x - out_x * p.st_x + p.dl_x - 1) / p.dl_x);
    bounds.x_max = min_i64(p.kw, (p.iw + p.pd_x - out_x * p.st_x + p.dl_x - 1) / p.dl_x);
    return bounds;
}

static inline int64_t calculate_input_coord(int64_t out_coord, int64_t kern_coord, int64_t stride, int64_t dilation, int64_t padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

struct whcn_layout {
    static inline int64_t input_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv_params & p) {
        return n * (p.ic * p.iw * p.ih) + c * p.iw * p.ih + y * p.iw + x;
    }

    static inline int64_t kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, const conv_params & p) {
        return c_out * (p.ic * p.kh * p.kw) + c_in * (p.kh * p.kw) + ky * p.kw + kx;
    }

    static inline int64_t output_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv_params & p) {
        return n * (p.oc * p.ow * p.oh) + c * p.ow * p.oh + y * p.ow + x;
    }

    static inline void unpack_indices(int64_t global_idx, const conv_params & p, int64_t & n, int64_t & c, int64_t & out_y, int64_t & out_x) {
        out_x = global_idx % p.ow;
        out_y = (global_idx / p.ow) % p.oh;
        c = (global_idx / (p.ow * p.oh)) % p.oc;
        n = global_idx / (p.ow * p.oh * p.oc);
    }
};

template <typename T>
inline float to_float(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else {
        return (float) value;
    }
}

template <typename T>
void conv2d_dispatch(sycl::queue & q, const float * input, const T * kernel, float * output, const conv_params & p) {
    const int64_t block = 256;
    const int64_t nblk = (p.total + block - 1) / block;

    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
            [=](sycl::nd_item<1> item) {
                const int64_t global_idx = item.get_global_id(0);
                if (global_idx >= p.total) {
                    return;
                }

                int64_t n;
                int64_t c_out;
                int64_t out_y;
                int64_t out_x;
                whcn_layout::unpack_indices(global_idx, p, n, c_out, out_y, out_x);

                float acc = 0.0f;
                const kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, p);

                for (int64_t c_in = 0; c_in < p.ic; ++c_in) {
                    for (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
                        const int64_t in_y = calculate_input_coord(out_y, ky, p.st_y, p.dl_y, p.pd_y);

                        for (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                            const int64_t in_x = calculate_input_coord(out_x, kx, p.st_x, p.dl_x, p.pd_x);
                            acc += input[whcn_layout::input_index(n, c_in, in_y, in_x, p)] *
                                   to_float(kernel[whcn_layout::kernel_index(c_out, c_in, ky, kx, p)]);
                        }
                    }
                }

                output[whcn_layout::output_index(n, c_out, out_y, out_x, p)] = acc;
            });
}
}

void ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(input->ne[2] == kernel->ne[2]);
    GGML_ASSERT(ggml_get_op_params_i32(dst, 6) == 0);

    conv_params p = {
        input->ne[0],
        input->ne[1],
        dst->ne[0],
        dst->ne[1],
        kernel->ne[0],
        kernel->ne[1],
        ggml_get_op_params_i32(dst, 0),
        ggml_get_op_params_i32(dst, 1),
        ggml_get_op_params_i32(dst, 2),
        ggml_get_op_params_i32(dst, 3),
        ggml_get_op_params_i32(dst, 4),
        ggml_get_op_params_i32(dst, 5),
        input->ne[2],
        kernel->ne[3],
        input->ne[3],
        input->ne[3] * kernel->ne[3] * dst->ne[1] * dst->ne[0],
    };

    sycl::queue & q = ctx.stream();
    if (kernel->type == GGML_TYPE_F32) {
        conv2d_dispatch(q, (const float *) input->data, (const float *) kernel->data, (float *) dst->data, p);
    } else {
        conv2d_dispatch(q, (const float *) input->data, (const sycl::half *) kernel->data, (float *) dst->data, p);
    }
}

bool ggml_sycl_supports_conv2d(const ggml_tensor * op) {
    return ggml_is_contiguous(op->src[0]) &&
           ggml_is_contiguous(op->src[1]) &&
           (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
           op->src[1]->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_get_op_params_i32(op, 6) == 0;
}
