#include "cross-entropy-loss.hpp"
#include "sumrows.hpp"

#define CEL_BLOCK_SIZE 256

// Forward kernel — one work-group (CEL_BLOCK_SIZE threads) per row.
// Three passes over nclasses:
//   1. Find max logit (block_reduce MAX)
//   2. Compute sum of exp(logit - max) -> log(sum)
//   3. Compute loss = -(logit - max - log_sum) * label, reduced / nrows
// When use_shared=true, logits are cached in local (shared) memory.
template <bool use_shared>
static void cross_entropy_loss_f32_sycl(
    const float * logits, const float * labels, float * dst,
    const int nclasses, const int nrows,
    sycl::queue & stream) {

    constexpr int block_size = CEL_BLOCK_SIZE;
    constexpr int num_warps = block_size / WARP_SIZE;

    const sycl::range<1> global(nrows * block_size);
    const sycl::range<1> local(block_size);

    stream.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> tmp_acc(use_shared ? nclasses : 0, cgh);
        sycl::local_accessor<float, 1> reduce_acc(num_warps, cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int row = item.get_group(0);
                const int tid = item.get_local_id(0);

                const float * row_logits = logits + (int64_t)row * nclasses;
                const float * row_labels = labels + (int64_t)row * nclasses;

                float * tmp = nullptr;
                if constexpr (use_shared) {
                    tmp = tmp_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
                }
                float * reduce_smem = reduce_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();

                // Pass 1: find max logit
                float max_logit = -INFINITY;
                for (int i = tid; i < nclasses; i += block_size) {
                    const float val = row_logits[i];
                    max_logit = sycl::fmax(max_logit, val);
                    if constexpr (use_shared) {
                        tmp[i] = val;
                    }
                }
                max_logit = block_reduce<block_reduce_method::MAX, block_size>(max_logit, reduce_smem, item);
                // Broadcast max from lane 0 to all threads via SLM
                if (tid == 0) { reduce_smem[0] = max_logit; }
                item.barrier(sycl::access::fence_space::local_space);
                max_logit = reduce_smem[0];
                item.barrier(sycl::access::fence_space::local_space);

                // Pass 2: sum of exp(logit - max)
                float sum = 0.0f;
                for (int i = tid; i < nclasses; i += block_size) {
                    const float logit_i = use_shared ? tmp[i] : row_logits[i];
                    sum += sycl::native::exp(logit_i - max_logit);
                }
                sum = block_reduce<block_reduce_method::SUM, block_size>(sum, reduce_smem, item);
                // Broadcast sum from lane 0 to all threads
                if (tid == 0) { reduce_smem[0] = sycl::native::log(sum); }
                item.barrier(sycl::access::fence_space::local_space);
                const float log_sum = reduce_smem[0];
                item.barrier(sycl::access::fence_space::local_space);

                // Pass 3: cross-entropy loss
                float loss = 0.0f;
                for (int i = tid; i < nclasses; i += block_size) {
                    const float logit_i = use_shared ? tmp[i] : row_logits[i];
                    loss += (logit_i - max_logit - log_sum) * row_labels[i];
                }
                loss = -block_reduce<block_reduce_method::SUM, block_size>(loss, reduce_smem, item) / (float)nrows;

                if (tid == 0) {
                    dst[row] = loss;
                }
            });
    });
}

// Backward kernel — one work-group (CEL_BLOCK_SIZE threads) per row.
// Two passes:
//   1. Find max logit, compute exp(logit - max) and sum
//   2. Output: (softmax(logit) - label) * grad / nrows
template <bool use_shared>
static void cross_entropy_loss_back_f32_sycl(
    const float * grad, const float * logits, const float * labels,
    float * dst, const int nclasses, const int nrows,
    sycl::queue & stream) {

    constexpr int block_size = CEL_BLOCK_SIZE;
    constexpr int num_warps = block_size / WARP_SIZE;

    const sycl::range<1> global(nrows * block_size);
    const sycl::range<1> local(block_size);

    stream.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> tmp_acc(use_shared ? nclasses : 0, cgh);
        sycl::local_accessor<float, 1> reduce_acc(num_warps, cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int row = item.get_group(0);
                const int tid = item.get_local_id(0);

                const float * row_logits = logits + (int64_t)row * nclasses;
                const float * row_labels = labels + (int64_t)row * nclasses;
                float       * row_dst    = dst    + (int64_t)row * nclasses;

                float * tmp = nullptr;
                if constexpr (use_shared) {
                    tmp = tmp_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
                }
                float * reduce_smem = reduce_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();

                // Pass 1: find max logit
                float maxval = -INFINITY;
                for (int i = tid; i < nclasses; i += block_size) {
                    const float val = row_logits[i];
                    maxval = sycl::fmax(maxval, val);
                    if constexpr (use_shared) {
                        tmp[i] = val;
                    }
                }
                maxval = block_reduce<block_reduce_method::MAX, block_size>(maxval, reduce_smem, item);
                // Broadcast max from lane 0 to all threads
                if (tid == 0) { reduce_smem[0] = maxval; }
                item.barrier(sycl::access::fence_space::local_space);
                maxval = reduce_smem[0];
                item.barrier(sycl::access::fence_space::local_space);

                // Compute exp(logit - max) and accumulate sum
                float sum = 0.0f;
                for (int i = tid; i < nclasses; i += block_size) {
                    const float val = sycl::native::exp((use_shared ? tmp[i] : row_logits[i]) - maxval);
                    sum += val;
                    if constexpr (use_shared) {
                        tmp[i] = val;
                    } else {
                        row_dst[i] = val;
                    }
                }
                sum = block_reduce<block_reduce_method::SUM, block_size>(sum, reduce_smem, item);
                // Broadcast sm_scale from lane 0 to all threads
                if (tid == 0) { reduce_smem[0] = 1.0f / sum; }
                item.barrier(sycl::access::fence_space::local_space);
                const float sm_scale = reduce_smem[0];
                item.barrier(sycl::access::fence_space::local_space);

                // Pass 2: output (softmax - label) * grad / nrows
                const float d_by_nrows = grad[0] / (float)nrows;
                for (int i = tid; i < nclasses; i += block_size) {
                    const float val = use_shared ? tmp[i] : row_dst[i];
                    row_dst[i] = (val * sm_scale - row_labels[i]) * d_by_nrows;
                }
            });
    });
}

void ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * src0 = dst->src[0];  // logits
    const ggml_tensor * src1 = dst->src[1];  // labels

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    sycl::queue & stream = *ctx.stream();

    // Allocate temporary per-row loss buffer
    ggml_sycl_pool_alloc<float> dst_tmp(ctx.pool(), nrows);

    const size_t nbytes_shared = ne00 * sizeof(float) + (CEL_BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    // 49152 = 48KB, conservative local memory limit (Xe2 has 64KB)
    if (nbytes_shared <= 49152) {
        cross_entropy_loss_f32_sycl<true>(src0_d, src1_d, dst_tmp.ptr, ne00, nrows, stream);
    } else {
        cross_entropy_loss_f32_sycl<false>(src0_d, src1_d, dst_tmp.ptr, ne00, nrows, stream);
    }

    // Sum all per-row losses into a single scalar
    reduce_rows_f32_sycl(dst_tmp.ptr, dst_d, nrows, 1, false, stream);
}

void ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    const ggml_tensor * grad  = dst->src[0];  // upstream gradient (scalar)
    const ggml_tensor * src0f = dst->src[1];  // logits
    const ggml_tensor * src1f = dst->src[2];  // labels

    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(src1f->type == GGML_TYPE_F32);
    GGML_ASSERT(grad->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne00  = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    const float * src1f_d = (const float *) src1f->data;
    float       * dst_d   = (float       *) dst->data;

    sycl::queue & stream = *ctx.stream();

    const size_t nbytes_shared = ne00 * sizeof(float) + (CEL_BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    // 49152 = 48KB, conservative local memory limit (Xe2 has 64KB)
    if (nbytes_shared <= 49152) {
        cross_entropy_loss_back_f32_sycl<true>(grad_d, src0f_d, src1f_d, dst_d, ne00, nrows, stream);
    } else {
        cross_entropy_loss_back_f32_sycl<false>(grad_d, src0f_d, src1f_d, dst_d, ne00, nrows, stream);
    }
}
