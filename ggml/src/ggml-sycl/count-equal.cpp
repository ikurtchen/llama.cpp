#include "count-equal.hpp"

#include <sycl/sycl.hpp>

namespace {

template <typename T>
static inline T ggml_sycl_work_group_sum(sycl::nd_item<1> item, sycl::local_accessor<T, 1> scratch, T value) {
    const sycl::sub_group sg = item.get_sub_group();
    const uint32_t lid       = item.get_local_id(0);
    const uint32_t sg_lid    = sg.get_local_id()[0];
    const uint32_t sg_size   = sg.get_local_range()[0];
    const uint32_t sg_id     = lid / sg_size;
    const uint32_t wg_size   = item.get_local_range()[0];
    const uint32_t sg_count  = (wg_size + sg_size - 1) / sg_size;

    const T sg_total = sycl::reduce_over_group(sg, value, sycl::plus<T>());
    if (sg_lid == 0) {
        scratch[sg_id] = sg_total;
    }
    sycl::group_barrier(item.get_group());

    if (sg_id == 0) {
        const T partial = lid < sg_count ? scratch[lid] : T(0);
        const T wg_total = sycl::reduce_over_group(sg, partial, sycl::plus<T>());
        if (sg_lid == 0) {
            scratch[0] = wg_total;
        }
    }
    sycl::group_barrier(item.get_group());

    return scratch[0];
}

static void count_equal_i32(sycl::queue & q, const int32_t * x, const int32_t * y, int64_t * dst, int64_t ne) {
    constexpr size_t wg = 256;
    const size_t ng = ((size_t) ne + wg - 1) / wg;

    SYCL_CHECK(q.memset(dst, 0, sizeof(int64_t)));

    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<int64_t, 1> scratch(sycl::range<1>(wg), h);

        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(ng * wg), sycl::range<1>(wg)), [=](sycl::nd_item<1> item) {
            const int64_t i = (int64_t) item.get_global_id(0);
            const int64_t value = (i < ne && x[i] == y[i]) ? 1 : 0;
            const int64_t total = ggml_sycl_work_group_sum(item, scratch, value);

            if (item.get_local_id(0) == 0) {
                sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(*dst).fetch_add(total);
            }
        });
    });
}

} // namespace

void ggml_sycl_op_count_equal(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);
    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_I64);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    count_equal_i32(
        ctx.stream(),
        (const int32_t *) src0->data,
        (const int32_t *) src1->data,
        (int64_t *) dst->data,
        ggml_nelements(src0));
}

bool ggml_sycl_supports_count_equal(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_COUNT_EQUAL) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    return src0 != nullptr &&
           src1 != nullptr &&
           src0->type == GGML_TYPE_I32 &&
           src1->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_I64 &&
           ggml_are_same_shape(src0, src1) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
}
