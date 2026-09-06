#include "cross-entropy-loss.hpp"

#include <sycl/sycl.hpp>

#include <cmath>

namespace {

static size_t ggml_sycl_cross_entropy_wg_size(sycl::queue & q, int64_t nclasses) {
    const size_t max_wg = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    size_t wg = 32;
    while (wg < (size_t) nclasses && wg < max_wg && wg < 256) {
        wg *= 2;
    }
    return std::min(wg, max_wg);
}

}

void ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * logits = dst->src[0];
    const ggml_tensor * labels = dst->src[1];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(labels->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(logits));
    GGML_ASSERT(ggml_is_contiguous(labels));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t nclasses = logits->ne[0];
    const int64_t nrows = ggml_nrows(logits);

    const float * logits_d = (const float *) logits->data;
    const float * labels_d = (const float *) labels->data;
    float * dst_d = (float *) dst->data;

    sycl::queue & q = ctx.stream();
    const size_t wg = ggml_sycl_cross_entropy_wg_size(q, nclasses);

    SYCL_CHECK(q.memset(dst_d, 0, sizeof(float)));
    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nrows * (int64_t) wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) {
                const int64_t row = item.get_group(0);
                const int64_t lid = item.get_local_id(0);
                const float * row_logits = logits_d + row * nclasses;
                const float * row_labels = labels_d + row * nclasses;

                float max_logit = -INFINITY;
                for (int64_t i = lid; i < nclasses; i += (int64_t) wg) {
                    max_logit = sycl::fmax(max_logit, row_logits[i]);
                }
                max_logit = sycl::reduce_over_group(item.get_group(), max_logit, sycl::maximum<float>());

                float sum = 0.0f;
                for (int64_t i = lid; i < nclasses; i += (int64_t) wg) {
                    sum += sycl::exp(row_logits[i] - max_logit);
                }
                sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
                const float log_sum = sycl::log(sum);

                float loss = 0.0f;
                for (int64_t i = lid; i < nclasses; i += (int64_t) wg) {
                    loss += (row_logits[i] - max_logit - log_sum) * row_labels[i];
                }
                loss = -sycl::reduce_over_group(item.get_group(), loss, sycl::plus<float>()) / (float) nrows;

                if (lid == 0) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                            dst_ref(dst_d[0]);
                    dst_ref.fetch_add(loss);
                }
            });
}

void ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad   = dst->src[0];
    const ggml_tensor * logits = dst->src[1];
    const ggml_tensor * labels = dst->src[2];

    GGML_ASSERT(grad->type == GGML_TYPE_F32);
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(labels->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(ggml_is_contiguous(logits));
    GGML_ASSERT(ggml_is_contiguous(labels));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(logits, labels));
    GGML_ASSERT(ggml_are_same_shape(logits, dst));

    const int64_t nclasses = logits->ne[0];
    const int64_t nrows = ggml_nrows(logits);

    const float * grad_d = (const float *) grad->data;
    const float * logits_d = (const float *) logits->data;
    const float * labels_d = (const float *) labels->data;
    float * dst_d = (float *) dst->data;

    sycl::queue & q = ctx.stream();
    const size_t wg = ggml_sycl_cross_entropy_wg_size(q, nclasses);

    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nrows * (int64_t) wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) {
                const int64_t row = item.get_group(0);
                const int64_t lid = item.get_local_id(0);

                const float * row_logits = logits_d + row * nclasses;
                const float * row_labels = labels_d + row * nclasses;
                float * row_dst = dst_d + row * nclasses;

                float max_logit = -INFINITY;
                for (int64_t i = lid; i < nclasses; i += (int64_t) wg) {
                    max_logit = sycl::fmax(max_logit, row_logits[i]);
                }
                max_logit = sycl::reduce_over_group(item.get_group(), max_logit, sycl::maximum<float>());

                float sum = 0.0f;
                for (int64_t i = lid; i < nclasses; i += (int64_t) wg) {
                    sum += sycl::exp(row_logits[i] - max_logit);
                }
                sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());

                const float scale = grad_d[0] / ((float) nrows * sum);
                for (int64_t i = lid; i < nclasses; i += (int64_t) wg) {
                    const float val = sycl::exp(row_logits[i] - max_logit);
                    row_dst[i] = val * scale - row_labels[i] * grad_d[0] / (float) nrows;
                }
            });
}

bool ggml_sycl_supports_cross_entropy_loss(const ggml_tensor * op) {
    return op->type == GGML_TYPE_F32 &&
           op->src[0]->type == GGML_TYPE_F32 &&
           op->src[1]->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(op->src[0]) &&
           ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op);
}

bool ggml_sycl_supports_cross_entropy_loss_back(const ggml_tensor * op) {
    return op->type == GGML_TYPE_F32 &&
           op->src[0]->type == GGML_TYPE_F32 &&
           op->src[1]->type == GGML_TYPE_F32 &&
           op->src[2]->type == GGML_TYPE_F32 &&
           ggml_is_scalar(op->src[0]) &&
           ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op->src[2]) &&
           ggml_is_contiguous(op) &&
           ggml_are_same_shape(op->src[1], op->src[2]) &&
           ggml_are_same_shape(op->src[1], op);
}
