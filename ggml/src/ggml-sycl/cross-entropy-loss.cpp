#include "cross-entropy-loss.hpp"

#include "common.hpp"

template <bool use_shared>
static void cross_entropy_loss_f32(
        const float * __restrict__ logits, const float * __restrict__ labels,
        float * __restrict__ dst, const int nclasses, const int k,
        const sycl::nd_item<3> & item_ct1, float * tmp) {

    const int tid = item_ct1.get_local_id(2);
    const int row = item_ct1.get_group(1);

    logits += int64_t(row) * nclasses;
    labels += int64_t(row) * nclasses;

    // Find maximum for softmax:
    float max_logit = -INFINITY;
    for (int i = tid; i < nclasses; i += WARP_SIZE) {
        const float val = logits[i];
        max_logit = sycl::fmax(max_logit, val);

        if (use_shared) {
            tmp[i] = val;
        }
    }
    max_logit = warp_reduce_max(max_logit, item_ct1);

    // Calculate log(sum(exp(logits - max))):
    float sum = 0.0f;
    for (int i = tid; i < nclasses; i += WARP_SIZE) {
        const float logit_i = use_shared ? tmp[i] : logits[i];
        sum += sycl::exp(logit_i - max_logit);
    }
    sum = warp_reduce_sum(sum, item_ct1);
    sum = sycl::log(sum);

    // log(exp(logits - max) / sum) = (logits - max) - log(sum)
    float loss = 0.0f;
    for (int i = tid; i < nclasses; i += WARP_SIZE) {
        const float logit_i = use_shared ? tmp[i] : logits[i];
        loss += (logit_i - max_logit - sum) * labels[i];
    }
    loss = -warp_reduce_sum(loss, item_ct1) / (float)k;

    if (tid != 0) {
        return;
    }

    dst[row] = loss;
}

template <bool use_shared>
static void cross_entropy_loss_back_f32(
        const float * __restrict__ grad, const float * __restrict__ logits,
        const float * __restrict__ labels, float * __restrict__ dst,
        const int nclasses, const int nrows,
        const sycl::nd_item<3> & item_ct1, float * tmp) {

    const int tid = item_ct1.get_local_id(2);
    const int row = item_ct1.get_group(1);

    logits += int64_t(row) * nclasses;
    labels += int64_t(row) * nclasses;
    dst    += int64_t(row) * nclasses;

    float maxval = -INFINITY;
    for (int i = tid; i < nclasses; i += WARP_SIZE) {
        const float val = logits[i];
        maxval = sycl::fmax(maxval, val);

        if (use_shared) {
            tmp[i] = val;
        }
    }
    maxval = warp_reduce_max(maxval, item_ct1);

    float sum = 0.0f;
    for (int i = tid; i < nclasses; i += WARP_SIZE) {
        const float val = sycl::exp((use_shared ? tmp[i] : logits[i]) - maxval);
        sum += val;

        if (use_shared) {
            tmp[i] = val;
        } else {
            dst[i] = val;
        }
    }
    sum = warp_reduce_sum(sum, item_ct1);
    const float sm_scale = 1.0f / sum;

    const float d_by_nrows = *grad / nrows;
    for (int i = tid; i < nclasses; i += WARP_SIZE) {
        const float val = use_shared ? tmp[i] : dst[i];
        dst[i] = (val * sm_scale - labels[i]) * d_by_nrows;
    }
}

// Simple sum kernel: sums ne floats from src into a single float in dst
static void k_sum_f32(const float * src, float * dst, const int ne,
                      const sycl::nd_item<3> & item_ct1) {
    const int tid = item_ct1.get_local_id(2);

    float sum = 0.0f;
    for (int i = tid; i < ne; i += WARP_SIZE) {
        sum += src[i];
    }
    sum = warp_reduce_sum(sum, item_ct1);

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

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
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
                    cross_entropy_loss_f32<true>(
                        src0_d, src1_d, dst_tmp_ptr, ne00, nrows,
                        item_ct1, tmp_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    } else {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32<false>(
                        src0_d, src1_d, dst_tmp_ptr, ne00, nrows,
                        item_ct1, nullptr);
                });
        });
    }

    // Sum per-row losses into final scalar:
    const sycl::range<3> sum_block_dims(1, 1, WARP_SIZE);
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

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t nbytes_shared = ne00 * sizeof(float);

    const size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    if (nbytes_shared <= smpbo) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> tmp_acc(sycl::range<1>(ne00), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32<true>(
                        grad_d, src0f_d, src1f_d, dst_d, ne00, nrows,
                        item_ct1, tmp_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    } else {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32<false>(
                        grad_d, src0f_d, src1f_d, dst_d, ne00, nrows,
                        item_ct1, nullptr);
                });
        });
    }
}
