#include "tsembd.hpp"

void ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int ne00       = (int) src0->ne[0];
    const int nb1        = (int) dst->nb[1];
    const int dim        = dst->op_params[0];
    const int max_period = dst->op_params[1];

    const int half_ceil    = (dim + 1) / 2;
    const int block_size   = SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    const int num_blocks_x = (half_ceil + block_size - 1) / block_size;

    sycl::range<2> global_range(ne00, num_blocks_x * block_size);
    sycl::range<2> local_range(1, block_size);

    stream->parallel_for(
        sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> item) {
            const int i = item.get_group(0);
            const int j = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

            float * embed_data = (float *)((char *) dst_d + i * nb1);

            const int half = dim / 2;
            if (dim % 2 != 0 && j == half) {
                embed_data[2 * half] = 0.0f;
            }

            if (j >= half) {
                return;
            }

            const float timestep = src0_d[i];
            const float freq = sycl::exp(-sycl::log((float) max_period) * j / half);
            const float arg  = timestep * freq;
            embed_data[j]        = sycl::cos(arg);
            embed_data[j + half] = sycl::sin(arg);
        });
}
