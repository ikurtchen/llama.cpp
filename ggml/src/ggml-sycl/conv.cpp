#include "conv.hpp"

void ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // weight
    const ggml_tensor * src1 = dst->src[1]; // input

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float       * dst_d  = (float *)dst->data;

    sycl::queue & stream = *ctx.stream();

    const int32_t * opts = (const int32_t *)dst->op_params;
    const int s0 = opts[0];

    // Extract tensor dimensions to local variables (never capture ggml_tensor* into device code)
    const int src0_ne0 = src0->ne[0];
    const int src0_ne1 = src0->ne[1];
    const int src0_ne2 = src0->ne[2];
    const int src1_ne0 = src1->ne[0];
    const int dst_ne0  = dst->ne[0];

    const int64_t output_size = ggml_nelements(dst);

    const int block_size = SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    const int num_blocks = (output_size + block_size - 1) / block_size;

    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * block_size, block_size),
        [=](sycl::nd_item<1> item) {
            const int global_index = item.get_global_id(0);
            if (global_index >= output_size) {
                return;
            }

            const int out_index = global_index / dst_ne0;

            float accumulator = 0;

            for (int c = 0; c < src0_ne2; c++) {
                const int idx = global_index % dst_ne0;

                const int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
                const int input_offset = src1_ne0 * c;

                for (int i = 0; i < src1_ne0; i++) {
                    if (!(idx >= i * s0 && idx < i * s0 + src0_ne0)) {
                        continue;
                    }
                    const int weight_idx = idx - i * s0;

                    const float kernel_weight = src0_d[kernel_offset + weight_idx];
                    const float input_value   = src1_d[input_offset + i];

                    accumulator += kernel_weight * input_value;
                }
            }
            dst_d[global_index] = accumulator;
        }
    );
}
