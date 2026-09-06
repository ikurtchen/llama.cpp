#include "conv-transpose-1d.hpp"

#include <sycl/sycl.hpp>

namespace {
void conv_transpose_1d_dispatch(
        sycl::queue & q,
        int s0,
        int p0,
        int d0,
        int src0_ne0,
        int src0_ne1,
        int src0_ne2,
        int src1_ne0,
        int src1_ne1,
        int dst_ne0,
        int dst_ne1,
        int64_t output_size,
        const float * src0,
        const float * src1,
        float * dst) {
    const int64_t block = 256;
    const int64_t nblk = (output_size + block - 1) / block;

    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
            [=](sycl::nd_item<1> item) {
                const int global_index = (int) item.get_global_id(0);
                if (global_index >= output_size) {
                    return;
                }

                const int out_t = global_index % dst_ne0;
                const int out_ch = (global_index / dst_ne0) % dst_ne1;
                const int plane = global_index / (dst_ne0 * dst_ne1);

                float accumulator = 0.0f;
                for (int c = 0; c < src0_ne2; ++c) {
                    const int kernel_offset = src0_ne0 * (out_ch + src0_ne1 * c);
                    const int input_offset = src1_ne0 * (c + src1_ne1 * plane);

                    for (int k = 0; k < src0_ne0; ++k) {
                        const int input_numer = out_t + p0 - k * d0;
                        if (input_numer < 0 || input_numer % s0 != 0) {
                            continue;
                        }

                        const int input_t = input_numer / s0;
                        if (input_t >= src1_ne0) {
                            continue;
                        }

                        accumulator += src0[kernel_offset + k] * src1[input_offset + input_t];
                    }
                }

                dst[global_index] = accumulator;
            });
}
}

void ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int s0 = ggml_get_op_params_i32(dst, 0);
    const int p0 = 0;
    const int d0 = 1;

    conv_transpose_1d_dispatch(
            ctx.stream(),
            s0,
            p0,
            d0,
            (int) src0->ne[0],
            (int) src0->ne[1],
            (int) src0->ne[2],
            (int) src1->ne[0],
            (int) src1->ne[1],
            (int) dst->ne[0],
            (int) dst->ne[1],
            ggml_nelements(dst),
            (const float *) src0->data,
            (const float *) src1->data,
            (float *) dst->data);
}

bool ggml_sycl_supports_conv_transpose_1d(const ggml_tensor * op) {
    return op->src[0]->type == GGML_TYPE_F32 &&
           op->src[1]->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(op->src[0]) &&
           ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op);
}
