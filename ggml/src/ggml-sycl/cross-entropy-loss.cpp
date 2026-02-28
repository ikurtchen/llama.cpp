#include "cross-entropy-loss.hpp"

#include "common.hpp"

// opt_task_010: Increase cross-entropy work-group from 16 to 256 for Intel Xe2
// Uses sycl::reduce_over_group for multi-sub-group reduction instead of warp_reduce

template <bool use_shared, int CE_WG_SIZE = 256>
static void cross_entropy_loss_f32(
        const float * __restrict__ logits, const float * __restrict__ labels,
        float * __restrict__ dst, const int nclasses, const int k,
        const sycl::nd_item<3> & item_ct1, float * tmp) {

    const int tid = item_ct1.get_local_id(2);
    const int row = item_ct1.get_group(1);
    const int wg_size = item_ct1.get_local_range(2);

    logits += int64_t(row) * nclasses;
    labels += int64_t(row) * nclasses;

    // Find maximum for softmax:
    float max_logit = -INFINITY;
    for (int i = tid; i < nclasses; i += wg_size) {
        const float val = logits[i];
        max_logit = sycl::fmax(max_logit, val);

        if (use_shared) {
            tmp[i] = val;
        }
    }
    // opt_027_001: Use sycl::reduce_over_group for multi-sub-group reduction
    max_logit = sycl::reduce_over_group(item_ct1.get_group(), max_logit, sycl::maximum<float>());

    // Calculate log(sum(exp(logits - max))):
    // opt_task_013: Use sycl::native::exp for faster computation
    // See: hw_spec, optimization_guide
    float sum = 0.0f;
    for (int i = tid; i < nclasses; i += wg_size) {
        const float logit_i = use_shared ? tmp[i] : logits[i];
        sum += sycl::native::exp(logit_i - max_logit);
    }
    // opt_task_013: Use sycl::native::log for faster computation
    // See: hw_spec, optimization_guide
    sum = sycl::reduce_over_group(item_ct1.get_group(), sum, sycl::plus<float>());
    sum = sycl::native::log(sum);

    // log(exp(logits - max) / sum) = (logits - max) - log(sum)
    float loss = 0.0f;
    for (int i = tid; i < nclasses; i += wg_size) {
        const float logit_i = use_shared ? tmp[i] : logits[i];
        loss += (logit_i - max_logit - sum) * labels[i];
    }
    // opt_027_001: Use sycl::reduce_over_group for multi-sub-group reduction
    loss = -sycl::reduce_over_group(item_ct1.get_group(), loss, sycl::plus<float>()) / (float)k;

    if (tid != 0) {
        return;
    }

    dst[row] = loss;
}

template <bool use_shared, int CE_WG_SIZE = 256>
static void cross_entropy_loss_back_f32(
        const float * __restrict__ grad, const float * __restrict__ logits,
        const float * __restrict__ labels, float * __restrict__ dst,
        const int nclasses, const int nrows,
        const sycl::nd_item<3> & item_ct1, float * tmp) {

    const int tid = item_ct1.get_local_id(2);
    const int row = item_ct1.get_group(1);
    const int wg_size = item_ct1.get_local_range(2);

    logits += int64_t(row) * nclasses;
    labels += int64_t(row) * nclasses;
    dst    += int64_t(row) * nclasses;

    float maxval = -INFINITY;
    for (int i = tid; i < nclasses; i += wg_size) {
        const float val = logits[i];
        maxval = sycl::fmax(maxval, val);

        if (use_shared) {
            tmp[i] = val;
        }
    }
    // opt_028_001: Use sycl::reduce_over_group for multi-sub-group reduction
    maxval = sycl::reduce_over_group(item_ct1.get_group(), maxval, sycl::maximum<float>());

    float sum = 0.0f;
    // opt_task_013: Use sycl::native::exp for faster computation
    // See: hw_spec, optimization_guide
    for (int i = tid; i < nclasses; i += wg_size) {
        const float val = sycl::native::exp((use_shared ? tmp[i] : logits[i]) - maxval);
        sum += val;

        if (use_shared) {
            tmp[i] = val;
        } else {
            dst[i] = val;
        }
    }
    // opt_028_001: Use sycl::reduce_over_group for multi-sub-group reduction
    sum = sycl::reduce_over_group(item_ct1.get_group(), sum, sycl::plus<float>());
    const float sm_scale = 1.0f / sum;

    // opt_028_003: Precompute grad/nrows as scalar argument instead of per-work-item
    const float d_by_nrows = *grad / nrows;
    for (int i = tid; i < nclasses; i += wg_size) {
        const float val = use_shared ? tmp[i] : dst[i];
        dst[i] = (val * sm_scale - labels[i]) * d_by_nrows;
    }
}

// opt_task_010: Optimized sum kernel with larger work-group
// Uses 256 threads instead of 16, with sycl::reduce_over_group
static void k_sum_f32(const float * src, float * dst, const int ne,
                      const sycl::nd_item<3> & item_ct1) {
    const int tid = item_ct1.get_local_id(2);
    const int wg_size = item_ct1.get_local_range(2);

    float sum = 0.0f;
    for (int i = tid; i < ne; i += wg_size) {
        sum += src[i];
    }
    // opt_027_002: Use sycl::reduce_over_group for multi-sub-group reduction
    sum = sycl::reduce_over_group(item_ct1.get_group(), sum, sycl::plus<float>());

    if (tid == 0) {
        dst[0] = sum;
    }
}

void ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    sycl::queue* stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    // opt_task_010: Increase work-group size from WARP_SIZE (16) to 256
    constexpr int CE_WG_SIZE = 256;
    const sycl::range<3> block_dims(1, 1, CE_WG_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t nbytes_shared = ne00 * sizeof(float);

    const size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    ggml_sycl_pool_alloc<float> dst_tmp(ctx.pool(), nrows);
    float * dst_tmp_ptr = dst_tmp.get();

    if (nbytes_shared <= smpbo) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> tmp_acc(sycl::range<1>(ne00), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32<true, CE_WG_SIZE>(
                        src0_d, src1_d, dst_tmp_ptr, ne00, nrows,
                        item_ct1, tmp_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    } else {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32<false, CE_WG_SIZE>(
                        src0_d, src1_d, dst_tmp_ptr, ne00, nrows,
                        item_ct1, nullptr);
                });
        });
    }

    // opt_027_002: Fix sum kernel to use 256 threads instead of 16
    // Keep single work-group for simplicity - each thread processes nrows/256 elements
    const sycl::range<3> sum_block_dims(1, 1, CE_WG_SIZE);
    const sycl::range<3> sum_block_nums(1, 1, 1);
    stream->parallel_for(
        sycl::nd_range<3>(sum_block_nums * sum_block_dims, sum_block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            k_sum_f32(dst_tmp_ptr, dst_d, nrows, item_ct1);
        });
}

void ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0];
    const ggml_tensor * src0f = dst->src[1];
    const ggml_tensor * src1f = dst->src[2];

    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(src1f->type == GGML_TYPE_F32);
    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f));
    GGML_ASSERT(ggml_are_same_shape(src0f, dst));

    const int64_t ne00  = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    const float * src1f_d = (const float *) src1f->data;
    float       * dst_d   = (float       *) dst->data;

    sycl::queue* stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    // opt_task_010: Increase work-group size from WARP_SIZE (16) to 256
    constexpr int CE_WG_SIZE = 256;
    const sycl::range<3> block_dims(1, 1, CE_WG_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t nbytes_shared = ne00 * sizeof(float);

    const size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    if (nbytes_shared <= smpbo) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> tmp_acc(sycl::range<1>(ne00), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32<true, CE_WG_SIZE>(
                        grad_d, src0f_d, src1f_d, dst_d, ne00, nrows,
                        item_ct1, tmp_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    } else {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32<false, CE_WG_SIZE>(
                        grad_d, src0f_d, src1f_d, dst_d, ne00, nrows,
                        item_ct1, nullptr);
                });
        });
    }
}
