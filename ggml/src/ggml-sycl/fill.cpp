#include "fill.hpp"

#include "common.hpp"

#define SYCL_FILL_BLOCK_SIZE 256

template <typename T>
static void fill_kernel(T * dst, const int64_t k, const T value,
                        const sycl::nd_item<1> & item_ct1) {
    const int64_t i = (int64_t)item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    if (i >= k) {
        return;
    }
    dst[i] = value;
}

void ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    float value;
    memcpy(&value, dst->op_params, sizeof(float));

    const int64_t k = ggml_nelements(dst);
    const int64_t num_blocks = ceil_div((size_t)k, (size_t)SYCL_FILL_BLOCK_SIZE);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    switch (dst->type) {
        case GGML_TYPE_F32:
            {
                float * dst_d = (float *)dst->data;
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_FILL_BLOCK_SIZE),
                                      sycl::range<1>(SYCL_FILL_BLOCK_SIZE)),
                    [=](sycl::nd_item<1> item_ct1) {
                        fill_kernel(dst_d, k, value, item_ct1);
                    });
            }
            break;
        case GGML_TYPE_F16:
            {
                sycl::half * dst_d = (sycl::half *)dst->data;
                sycl::half value_f16 = sycl::vec<float, 1>(value)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_FILL_BLOCK_SIZE),
                                      sycl::range<1>(SYCL_FILL_BLOCK_SIZE)),
                    [=](sycl::nd_item<1> item_ct1) {
                        fill_kernel(dst_d, k, value_f16, item_ct1);
                    });
            }
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
