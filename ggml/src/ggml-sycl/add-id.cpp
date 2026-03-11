#include "add-id.hpp"

#include <algorithm>

void ggml_sycl_add_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb20 == sizeof(int32_t));

    // Extract device pointers before lambda capture — never capture ggml_tensor* into device code
    const float   * src0_d = (const float   *) src0->data;
    const float   * src1_d = (const float   *) src1->data;
    const int32_t * src2_d = (const int32_t *) src2->data;
    float         * dst_d  = (float         *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    // Use hardware max work-group size (1024 on dGPU, may be smaller on iGPU)
    const unsigned int max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
    const int threads = std::min(static_cast<unsigned int>(ne00), max_wg);

    // SYCL nd_range<2>:
    //   dimension 0  → ne01 work-groups of size `threads`  (maps to blockIdx.x / threadIdx.x)
    //   dimension 1  → ne02 work-groups of size 1          (maps to blockIdx.y)
    const sycl::range<2> global_range(ne01 * threads, ne02);
    const sycl::range<2> local_range(threads, 1);

    // Capture the byte-stride values needed to compute row addresses on the device.
    const size_t k_nb01 = nb01;
    const size_t k_nb02 = nb02;
    const size_t k_nb11 = nb11;
    const size_t k_nb21 = nb21;
    const int64_t k_ne0 = ne0;
    const int64_t k_ne1 = ne1;

    stream->parallel_for(
        sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> item) {
            // Work-group indices — equivalent to CUDA blockIdx
            const int64_t i1 = item.get_group(0);  // blockIdx.x
            const int64_t i2 = item.get_group(1);  // blockIdx.y

            // Look up which src1 row to use for this (i1, i2) pair
            const int i11 = *(const int32_t *)((const char *) src2_d
                             + i1 * sizeof(int32_t) + i2 * k_nb21);

            // Compute contiguous dst / src0 byte strides (dst is always contiguous f32)
            const size_t dst_nb1 = k_ne0 * sizeof(float);
            const size_t dst_nb2 = k_ne1 * dst_nb1;

            float       * dst_row  = (float       *)((char       *) dst_d  + i1 * dst_nb1 + i2 * dst_nb2);
            const float * src0_row = (const float *)((const char *) src0_d + i1 * k_nb01  + i2 * k_nb02);
            const float * src1_row = (const float *)((const char *) src1_d + i11 * k_nb11);

            // Strided loop over columns — equivalent to CUDA threadIdx.x / blockDim.x
            const int64_t local_id    = item.get_local_id(0);
            const int64_t local_range = item.get_local_range(0);
            for (int64_t i0 = local_id; i0 < k_ne0; i0 += local_range) {
                dst_row[i0] = src0_row[i0] + src1_row[i0];
            }
        });
}
