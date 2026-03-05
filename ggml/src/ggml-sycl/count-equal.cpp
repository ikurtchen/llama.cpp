#include "count-equal.hpp"

#include <cstdint>

template <typename T>
static void count_equal_sycl(const T * x, const T * y, int64_t * dst,
                              const int64_t dk, const int64_t k,
                              const int nblocks, sycl::queue & stream) {
    const sycl::range<1> global_range(nblocks * WARP_SIZE);
    const sycl::range<1> local_range(WARP_SIZE);

    stream.submit([&](sycl::handler & h) {
        h.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int64_t i0 = (int64_t)item.get_group(0) * dk;
                const int64_t i1 = (i0 + dk < k) ? (i0 + dk) : k;

                int nequal = 0;

                for (int64_t i = i0 + item.get_local_id(0); i < i1; i += WARP_SIZE) {
                    const T xi = x[i];
                    const T yi = y[i];
                    nequal += xi == yi;
                }

                nequal = warp_reduce_sum(nequal);

                if (item.get_local_id(0) != 0) {
                    return;
                }

                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    dst_atomic(*reinterpret_cast<int *>(dst));
                dst_atomic.fetch_add(nequal);
            });
    });
}

void ggml_sycl_count_equal(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    // Extract device pointers before kernel lambda
    int64_t * dst_d = (int64_t *) dst->data;

    sycl::queue & stream = *(ctx.stream());

    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");

    const int nsm = 64; // reasonable default for SYCL devices
    const int64_t dne = GGML_PAD((ne + 4*nsm - 1) / (4*nsm), SYCL_COUNT_EQUAL_CHUNK_SIZE);
    const int nblocks = (int)std::min((int64_t)4*nsm, (ne + SYCL_COUNT_EQUAL_CHUNK_SIZE - 1) / SYCL_COUNT_EQUAL_CHUNK_SIZE);

    stream.memset(dst_d, 0, ggml_nbytes(dst));

    switch (src0->type) {
        case GGML_TYPE_I32: {
            const int * src0_d = (const int *) src0->data;
            const int * src1_d = (const int *) src1->data;
            count_equal_sycl(src0_d, src1_d, dst_d, dne, ne, nblocks, stream);
        } break;
        default:
            GGML_ASSERT(false);
            break;
    }
}
