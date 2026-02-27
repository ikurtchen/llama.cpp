#include "diag.hpp"

#include "common.hpp"

#define SYCL_DIAG_BLOCK_SIZE 256

template <typename T>
static void diag_kernel(T * __restrict__ dst,
                        const T * __restrict__ src,
                        const int64_t ne0,
                        const int64_t ne1,
                        const int64_t ne2,
                        const int64_t ne3,
                        const int64_t total_elements,
                        const sycl::nd_item<1> & item_ct1) {
    const int64_t global_idx = (int64_t)item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);

    if (global_idx >= total_elements) {
        return;
    }

    const int64_t i0 = global_idx % ne0;
    const int64_t i1 = (global_idx / ne0) % ne1;
    const int64_t i2 = (global_idx / (ne0 * ne1)) % ne2;
    const int64_t i3 = global_idx / (ne0 * ne1 * ne2);

    const int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    if (i0 == i1) {
        const int64_t batch_idx = i3 * ne2 + i2;
        const int64_t src_idx   = batch_idx * ne0 + i0;
        dst[dst_idx]            = src[src_idx];
    } else {
        dst[dst_idx] = static_cast<T>(0);
    }
    GGML_UNUSED(ne3);
}

void ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    void *       dst_d  = dst->data;
    const void * src0_d = src0->data;

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    const int64_t n_elems    = ggml_nelements(dst);
    const int64_t num_blocks = ceil_div((size_t)n_elems, (size_t)SYCL_DIAG_BLOCK_SIZE);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    switch (dst->type) {
        case GGML_TYPE_F32:
            {
                float *       dst_f  = (float *)dst_d;
                const float * src0_f = (const float *)src0_d;
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_DIAG_BLOCK_SIZE),
                                      sycl::range<1>(SYCL_DIAG_BLOCK_SIZE)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        diag_kernel(dst_f, src0_f, ne0, ne1, ne2, ne3, n_elems, item_ct1);
                    });
            }
            break;
        case GGML_TYPE_F16:
            {
                sycl::half *       dst_f16  = (sycl::half *)dst_d;
                const sycl::half * src0_f16 = (const sycl::half *)src0_d;
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_DIAG_BLOCK_SIZE),
                                      sycl::range<1>(SYCL_DIAG_BLOCK_SIZE)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        diag_kernel(dst_f16, src0_f16, ne0, ne1, ne2, ne3, n_elems, item_ct1);
                    });
            }
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
