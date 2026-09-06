#include "mmid.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>

void ggml_sycl_launch_mm_ids_helper(
        ggml_backend_sycl_context & ctx,
        const int32_t * ids,
        int32_t * ids_src1,
        int32_t * ids_dst,
        int32_t * expert_bounds,
        int n_experts,
        int n_tokens,
        int n_expert_used,
        int nchannels_y,
        int si1,
        int sis1,
        bool write_inverse) {
    GGML_ASSERT(ids != nullptr);
    GGML_ASSERT(ids_src1 != nullptr);
    GGML_ASSERT(ids_dst != nullptr);
    GGML_ASSERT(expert_bounds != nullptr);
    GGML_ASSERT(n_experts >= 0);
    GGML_ASSERT(n_tokens >= 0);
    GGML_ASSERT(n_expert_used >= 0);

    sycl::queue & q = ctx.stream();
    constexpr size_t wg_size = 32;
    const size_t n_groups = std::max<size_t>((size_t) n_experts, 1);

    q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(n_groups * wg_size), sycl::range<1>(wg_size)),
        [=](sycl::nd_item<1> item) {
            const int expert = (int) item.get_group_linear_id();
            if (expert >= n_experts || item.get_local_linear_id() != 0) {
                return;
            }

            int nex_prev = 0;
            int it_compact = 0;

            for (int it = 0; it < n_tokens; ++it) {
                int iex_used = -1;
                for (int iex = 0; iex < n_expert_used; ++iex) {
                    const int expert_used = ids[it * si1 + iex];
                    nex_prev += expert_used < expert;
                    if (expert_used == expert) {
                        iex_used = iex;
                    }
                }
                if (iex_used != -1) {
                    ++it_compact;
                }
            }

            int compact_idx = 0;
            for (int it = 0; it < n_tokens; ++it) {
                int iex_used = -1;
                for (int iex = 0; iex < n_expert_used; ++iex) {
                    if (ids[it * si1 + iex] == expert) {
                        iex_used = iex;
                        break;
                    }
                }
                if (iex_used == -1) {
                    continue;
                }

                const int dst_idx = nex_prev + compact_idx;
                ids_dst[dst_idx] = it * n_expert_used + iex_used;
                if (write_inverse) {
                    ids_src1[it * n_expert_used + iex_used] = dst_idx;
                } else {
                    ids_src1[dst_idx] = it * sis1 + iex_used % nchannels_y;
                }
                ++compact_idx;
            }

            expert_bounds[expert] = nex_prev;
            if (expert == n_experts - 1) {
                expert_bounds[n_experts] = nex_prev + it_compact;
            }
        });

    SYCL_CHECK(q.wait_and_throw());
}
