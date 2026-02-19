#include "mmvq.hpp"

#include "ggml.h"
#include "common.hpp"
#include "quants.hpp"
#include "vecdotq.hpp"

template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_reorder(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                  const int ncols, const int nrows, const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const auto         bx_offset      = block_type::get_block_offset(ibx, nblocks);
        const auto         d_offset       = block_type::get_d_offset(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        const int iby = i * block_type::block_to_q8_1_ratio();
        const int8_t* q8_1_quant_ptr = (const int8_t*)vy + iby * QK8_1;
        const sycl::half2* q8_1_ds_ptr = (const sycl::half2*)((const char*)vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

// Optimized MMVQ kernel with increased work-group size (nwarps sub-groups)
// This improves memory latency hiding by using more threads per work-group.
// On Intel Xe2 (SIMD16), this uses 4 sub-groups (64 threads) instead of 1 (16 threads).
// Each sub-group computes a partial sum over a subset of blocks_per_row.
// SLM is used for inter-sub-group reduction before final write.
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl, int nwarps = 4>
static void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                          const int ncols, const int nrows, const sycl::nd_item<3> & item_ct1,
                          float * slm_partial_sums) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    assert(blocks_per_warp > 0);

    const int thread_id = item_ct1.get_local_id(2);
    const int sg_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = sg_id * blocks_per_warp + lane_id / (qi / vdr); i < blocks_per_row; i += nwarps * blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;

        const int iby = i * (qk / QK8_1);

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (lane_id % (qi / vdr));

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane_id == 0) {
        slm_partial_sums[sg_id] = tmp;
    }

    sycl::group_barrier(item_ct1.get_group());

    if (sg_id == 0) {
        float sum = slm_partial_sums[0];
#pragma unroll
        for (int i = 1; i < nwarps; ++i) {
            sum += slm_partial_sums[i];
        }

        if (lane_id == 0) {
            dst[row] = sum;
        }
    }
}

// Optimized batched MMVQ kernel with increased work-group size (nwarps sub-groups)
// This improves memory latency hiding by using more threads per work-group.
// On Intel Xe2 (SIMD16), this uses 4 sub-groups (64 threads) instead of 1 (16 threads).
// Each sub-group computes a partial sum over a subset of blocks_per_row.
// SLM is used for inter-sub-group reduction before final write.
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl, int ncols_dst, int nwarps = 4>
static void mul_mat_vec_q_batched(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                   const int ncols, const int nrows, const int64_t stride_y, const int64_t stride_dst,
                                   const sycl::nd_item<3> & item_ct1, float * slm_partial_sums) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    assert(blocks_per_warp > 0);

    const int thread_id = item_ct1.get_local_id(2);
    const int sg_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    float tmp[ncols_dst] = {0.0f};

    const block_q_t * x = (const block_q_t *) vx;

    for (int i = sg_id * blocks_per_warp + lane_id / (qi / vdr); i < blocks_per_row; i += nwarps * blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iby = i * (qk / QK8_1);

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (lane_id % (qi / vdr));

#pragma unroll
            for (int col = 0; col < ncols_dst; ++col) {
                const block_q8_1 * y_col = (const block_q8_1 *)((const char*)vy + col * stride_y);
                tmp[col] += vec_dot_q_sycl(&x[ibx], &y_col[iby], iqs);
            }
        }
    }

#pragma unroll
    for (int col = 0; col < ncols_dst; ++col) {
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            tmp[col] += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp[col], mask);
        }
    }

    if (lane_id == 0) {
#pragma unroll
        for (int col = 0; col < ncols_dst; ++col) {
            slm_partial_sums[col * nwarps + sg_id] = tmp[col];
        }
    }

    sycl::group_barrier(item_ct1.get_group());

    if (sg_id == 0 && lane_id == 0) {
#pragma unroll
        for (int col = 0; col < ncols_dst; ++col) {
            float sum = slm_partial_sums[col * nwarps + 0];
#pragma unroll
            for (int i = 1; i < nwarps; ++i) {
                sum += slm_partial_sums[col * nwarps + i];
            }
            dst[row + col * stride_dst] = sum;
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, int nwarps = 4>
static void mul_mat_vec_q_iq2_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1,
                                       float * slm_partial_sums) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

    const int thread_id = item_ct1.get_local_id(2);
    const int sg_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = sg_id * blocks_per_warp + lane_id / (qi / vdr); i < blocks_per_row;
         i += nwarps * blocks_per_warp) {
        const int ibx = row*blocks_per_row + i;

        const int iby = i * (qk/QK8_1);

        const int iqs =
            vdr *
            (lane_id %
             (qi / vdr));

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane_id == 0) {
        slm_partial_sums[sg_id] = tmp;
    }

    sycl::group_barrier(item_ct1.get_group());

    if (sg_id == 0) {
        float sum = slm_partial_sums[0];
#pragma unroll
        for (int i = 1; i < nwarps; ++i) {
            sum += slm_partial_sums[i];
        }
        if (lane_id == 0) {
            dst[row] = sum;
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, int nwarps = 4>
static void mul_mat_vec_q_iq2_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1,
                                      float * slm_partial_sums) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

    const int thread_id = item_ct1.get_local_id(2);
    const int sg_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = sg_id * blocks_per_warp + lane_id / (qi / vdr); i < blocks_per_row;
         i += nwarps * blocks_per_warp) {
        const int ibx = row*blocks_per_row + i;

        const int iby = i * (qk/QK8_1);

        const int iqs =
            vdr *
            (lane_id %
             (qi / vdr));

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid, ksigns64);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane_id == 0) {
        slm_partial_sums[sg_id] = tmp;
    }

    sycl::group_barrier(item_ct1.get_group());

    if (sg_id == 0) {
        float sum = slm_partial_sums[0];
#pragma unroll
        for (int i = 1; i < nwarps; ++i) {
            sum += slm_partial_sums[i];
        }
        if (lane_id == 0) {
            dst[row] = sum;
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, int nwarps = 4>
static void mul_mat_vec_q_iq2_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1,
                                     float * slm_partial_sums) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

    const int thread_id = item_ct1.get_local_id(2);
    const int sg_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = sg_id * blocks_per_warp + lane_id / (qi / vdr); i < blocks_per_row;
         i += nwarps * blocks_per_warp) {
        const int ibx = row*blocks_per_row + i;

        const int iby = i * (qk/QK8_1);

        const int iqs =
            vdr *
            (lane_id %
             (qi / vdr));

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane_id == 0) {
        slm_partial_sums[sg_id] = tmp;
    }

    sycl::group_barrier(item_ct1.get_group());

    if (sg_id == 0) {
        float sum = slm_partial_sums[0];
#pragma unroll
        for (int i = 1; i < nwarps; ++i) {
            sum += slm_partial_sums[i];
        }
        if (lane_id == 0) {
            dst[row] = sum;
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, int nwarps = 4>
static void mul_mat_vec_q_iq3_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1,
                                       float * slm_partial_sums) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

    const int thread_id = item_ct1.get_local_id(2);
    const int sg_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = sg_id * blocks_per_warp + lane_id / (qi / vdr); i < blocks_per_row;
         i += nwarps * blocks_per_warp) {
        const int ibx = row*blocks_per_row + i;

        const int iby = i * (qk/QK8_1);

        const int iqs =
            vdr *
            (lane_id %
             (qi / vdr));

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid, ksigns64);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane_id == 0) {
        slm_partial_sums[sg_id] = tmp;
    }

    sycl::group_barrier(item_ct1.get_group());

    if (sg_id == 0) {
        float sum = slm_partial_sums[0];
#pragma unroll
        for (int i = 1; i < nwarps; ++i) {
            sum += slm_partial_sums[i];
        }
        if (lane_id == 0) {
            dst[row] = sum;
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, int nwarps = 4>
static void mul_mat_vec_q_iq3_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1,
                                     float * slm_partial_sums) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

    const int thread_id = item_ct1.get_local_id(2);
    const int sg_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = sg_id * blocks_per_warp + lane_id / (qi / vdr); i < blocks_per_row;
         i += nwarps * blocks_per_warp) {
        const int ibx = row*blocks_per_row + i;

        const int iby = i * (qk/QK8_1);

        const int iqs =
            vdr *
            (lane_id %
             (qi / vdr));

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane_id == 0) {
        slm_partial_sums[sg_id] = tmp;
    }

    sycl::group_barrier(item_ct1.get_group());

    if (sg_id == 0) {
        float sum = slm_partial_sums[0];
#pragma unroll
        for (int i = 1; i < nwarps; ++i) {
            sum += slm_partial_sums[i];
        }
        if (lane_id == 0) {
            dst[row] = sum;
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, int nwarps = 4>
static void mul_mat_vec_q_iq1_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1,
                                     float * slm_partial_sums) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

    const int thread_id = item_ct1.get_local_id(2);
    const int sg_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = sg_id * blocks_per_warp + lane_id / (qi / vdr); i < blocks_per_row;
         i += nwarps * blocks_per_warp) {
        const int ibx = row*blocks_per_row + i;

        const int iby = i * (qk/QK8_1);

        const int iqs =
            vdr *
            (lane_id %
             (qi / vdr));

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane_id == 0) {
        slm_partial_sums[sg_id] = tmp;
    }

    sycl::group_barrier(item_ct1.get_group());

    if (sg_id == 0) {
        float sum = slm_partial_sums[0];
#pragma unroll
        for (int i = 1; i < nwarps; ++i) {
            sum += slm_partial_sums[i];
        }
        if (lane_id == 0) {
            dst[row] = sum;
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, int nwarps = 4>
static void mul_mat_vec_q_iq1_m_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1,
                                     float * slm_partial_sums) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

    const int thread_id = item_ct1.get_local_id(2);
    const int sg_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = sg_id * blocks_per_warp + lane_id / (qi / vdr); i < blocks_per_row;
         i += nwarps * blocks_per_warp) {
        const int ibx = row*blocks_per_row + i;

        const int iby = i * (qk/QK8_1);

        const int iqs =
            vdr *
            (lane_id %
             (qi / vdr));

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane_id == 0) {
        slm_partial_sums[sg_id] = tmp;
    }

    sycl::group_barrier(item_ct1.get_group());

    if (sg_id == 0) {
        float sum = slm_partial_sums[0];
#pragma unroll
        for (int i = 1; i < nwarps; ++i) {
            sum += slm_partial_sums[i];
        }
        if (lane_id == 0) {
            dst[row] = sum;
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, int nwarps = 4>
static void mul_mat_vec_q_iq4_nl_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1,
                                      float * slm_partial_sums) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

    const int thread_id = item_ct1.get_local_id(2);
    const int sg_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = sg_id * blocks_per_warp + lane_id / (qi / vdr); i < blocks_per_row;
         i += nwarps * blocks_per_warp) {
        const int ibx = row*blocks_per_row + i;

        const int iby = i * (qk/QK8_1);

        const int iqs =
            vdr *
            (lane_id %
             (qi / vdr));

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane_id == 0) {
        slm_partial_sums[sg_id] = tmp;
    }

    sycl::group_barrier(item_ct1.get_group());

    if (sg_id == 0) {
        float sum = slm_partial_sums[0];
#pragma unroll
        for (int i = 1; i < nwarps; ++i) {
            sum += slm_partial_sums[i];
        }
        if (lane_id == 0) {
            dst[row] = sum;
        }
    }
}


template <int qk, int qi, typename block_q_t, int vdr, int nwarps = 4>
static void mul_mat_vec_q_iq4_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1,
                                      float * slm_partial_sums) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

    const int thread_id = item_ct1.get_local_id(2);
    const int sg_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = sg_id * blocks_per_warp + lane_id / (qi / vdr); i < blocks_per_row;
         i += nwarps * blocks_per_warp) {
        const int ibx = row*blocks_per_row + i;

        const int iby = i * (qk/QK8_1);

        const int iqs =
            vdr *
            (lane_id %
             (qi / vdr));

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane_id == 0) {
        slm_partial_sums[sg_id] = tmp;
    }

    sycl::group_barrier(item_ct1.get_group());

    if (sg_id == 0) {
        float sum = slm_partial_sums[0];
#pragma unroll
        for (int i = 1; i < nwarps; ++i) {
            sum += slm_partial_sums[i];
        }
        if (lane_id == 0) {
            dst[row] = sum;
        }
    }
}

static void reorder_mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE));
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

static void mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, nwarps>(
                                 vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                         });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q4_0_q8_1_batched_sycl(const void * vx, const void * vy, float * dst,
                                                const int ncols, const int nrows,
                                                const int64_t stride_y, const int64_t stride_dst,
                                                dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(ncols_dst * nwarps, cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_batched<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, ncols_dst, nwarps>(
                                 vx, vy, dst, ncols, nrows, stride_y, stride_dst, item_ct1, get_pointer(slm_acc));
                         });
    });
}

static void mul_mat_vec_q4_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK4_0, QI4_1, block_q4_1,
                                  VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q4_1_q8_1_batched_sycl(const void * vx, const void * vy, float * dst,
                                                const int ncols, const int nrows,
                                                const int64_t stride_y, const int64_t stride_dst,
                                                dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(ncols_dst * nwarps, cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_batched<QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, ncols_dst, nwarps>(
                                 vx, vy, dst, ncols, nrows, stride_y, stride_dst, item_ct1, get_pointer(slm_acc));
                         });
    });
}

static void mul_mat_vec_mxfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, nwarps>(
                                 vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                         });
    });
}

template <int ncols_dst>
static void mul_mat_vec_mxfp4_q8_1_batched_sycl(const void * vx, const void * vy, float * dst,
                                                 const int ncols, const int nrows,
                                                 const int64_t stride_y, const int64_t stride_dst,
                                                 dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(ncols_dst * nwarps, cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_batched<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, ncols_dst, nwarps>(
                                 vx, vy, dst, ncols, nrows, stride_y, stride_dst, item_ct1, get_pointer(slm_acc));
                         });
    });
}


static void mul_mat_vec_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK5_0, QI5_0, block_q5_0,
                                  VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q5_0_q8_1_batched_sycl(const void * vx, const void * vy, float * dst,
                                                const int ncols, const int nrows,
                                                const int64_t stride_y, const int64_t stride_dst,
                                                dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(ncols_dst * nwarps, cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_batched<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, ncols_dst, nwarps>(
                                 vx, vy, dst, ncols, nrows, stride_y, stride_dst, item_ct1, get_pointer(slm_acc));
                         });
    });
}

static void mul_mat_vec_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK5_1, QI5_1, block_q5_1,
                                  VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q5_1_q8_1_batched_sycl(const void * vx, const void * vy, float * dst,
                                                const int ncols, const int nrows,
                                                const int64_t stride_y, const int64_t stride_dst,
                                                dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(ncols_dst * nwarps, cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_batched<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, ncols_dst, nwarps>(
                                 vx, vy, dst, ncols, nrows, stride_y, stride_dst, item_ct1, get_pointer(slm_acc));
                         });
    });
}

static void mul_mat_vec_q8_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK8_0, QI8_0, block_q8_0,
                                  VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q8_0_q8_1_batched_sycl(const void * vx, const void * vy, float * dst,
                                                const int ncols, const int nrows,
                                                const int64_t stride_y, const int64_t stride_dst,
                                                dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(ncols_dst * nwarps, cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_batched<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, ncols_dst, nwarps>(
                                 vx, vy, dst, ncols, nrows, stride_y, stride_dst, item_ct1, get_pointer(slm_acc));
                         });
    });
}

static void mul_mat_vec_q2_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK_K, QI2_K, block_q2_K,
                                  VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q2_K_q8_1_batched_sycl(const void * vx, const void * vy, float * dst,
                                                const int ncols, const int nrows,
                                                const int64_t stride_y, const int64_t stride_dst,
                                                dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(ncols_dst * nwarps, cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_batched<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, ncols_dst, nwarps>(
                                 vx, vy, dst, ncols, nrows, stride_y, stride_dst, item_ct1, get_pointer(slm_acc));
                         });
    });
}

static void mul_mat_vec_q3_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK_K, QI3_K, block_q3_K,
                                  VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q3_K_q8_1_batched_sycl(const void * vx, const void * vy, float * dst,
                                                const int ncols, const int nrows,
                                                const int64_t stride_y, const int64_t stride_dst,
                                                dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(ncols_dst * nwarps, cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_batched<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, ncols_dst, nwarps>(
                                 vx, vy, dst, ncols, nrows, stride_y, stride_dst, item_ct1, get_pointer(slm_acc));
                         });
    });
}

static void mul_mat_vec_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
                                  VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q4_K_q8_1_batched_sycl(const void * vx, const void * vy, float * dst,
                                                const int ncols, const int nrows,
                                                const int64_t stride_y, const int64_t stride_dst,
                                                dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(ncols_dst * nwarps, cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_batched<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1, ncols_dst, nwarps>(
                                 vx, vy, dst, ncols, nrows, stride_y, stride_dst, item_ct1, get_pointer(slm_acc));
                         });
    });
}

static void reorder_mul_mat_vec_q4_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}


static void mul_mat_vec_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK_K, QI5_K, block_q5_K,
                                  VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q5_K_q8_1_batched_sycl(const void * vx, const void * vy, float * dst,
                                                const int ncols, const int nrows,
                                                const int64_t stride_y, const int64_t stride_dst,
                                                dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(ncols_dst * nwarps, cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_batched<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1, ncols_dst, nwarps>(
                                 vx, vy, dst, ncols, nrows, stride_y, stride_dst, item_ct1, get_pointer(slm_acc));
                         });
    });
}

static void reorder_mul_mat_vec_q6_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}
static void mul_mat_vec_q6_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK_K, QI6_K, block_q6_K,
                                  VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q6_K_q8_1_batched_sycl(const void * vx, const void * vy, float * dst,
                                                const int ncols, const int nrows,
                                                const int64_t stride_y, const int64_t stride_dst,
                                                dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(ncols_dst * nwarps, cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_batched<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1, ncols_dst, nwarps>(
                                 vx, vy, dst, ncols, nrows, stride_y, stride_dst, item_ct1, get_pointer(slm_acc));
                         });
    });
}


static void mul_mat_vec_iq2_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, block_iq2_xxs, 1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

static void mul_mat_vec_iq2_xs_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, block_iq2_xs, 1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

static void mul_mat_vec_iq2_s_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, block_iq2_s, 1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

static void mul_mat_vec_iq3_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, block_iq3_xxs, 1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

static void mul_mat_vec_iq3_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, block_iq3_s, 1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

static void mul_mat_vec_iq1_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, block_iq1_s, 1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

static void mul_mat_vec_iq1_m_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, block_iq1_m, 1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

static void mul_mat_vec_iq4_nl_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, block_iq4_nl, 2, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

static void mul_mat_vec_iq4_xs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    constexpr int nwarps = 4;
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, nwarps * WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_acc(nwarps, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, block_iq4_xs, 1, nwarps>(
                        vx, vy, dst, ncols, nrows, item_ct1, get_pointer(slm_acc));
                });
    });
}

void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_col_size,
                                const dpct::queue_ptr & stream) {
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne00     = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const int64_t stride_y   = src1_padded_col_size * q8_1_ts / q8_1_bs;
    const int64_t stride_dst = dst->ne[0];

    bool use_reorder = false;
    if ((ggml_tensor_extra_gpu *) dst->src[0]->extra) {
        use_reorder = ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
    }

#define CALL_BATCHED(func, ncols_this, vx, vy, dst, ncols, nrows, stride_y, stride_dst, stream) \
    do { \
        const int _ncols = (ncols_this); \
        if (_ncols == 1) func<1>(vx, vy, dst, ncols, nrows, stride_y, stride_dst, stream); \
        else if (_ncols == 2) func<2>(vx, vy, dst, ncols, nrows, stride_y, stride_dst, stream); \
        else if (_ncols <= 4) func<4>(vx, vy, dst, ncols, nrows, stride_y, stride_dst, stream); \
        else func<8>(vx, vy, dst, ncols, nrows, stride_y, stride_dst, stream); \
    } while (0)

    int col_offset = 0;
    while (col_offset < src1_ncols) {
        int ncols_this = std::min((int64_t)MMVQ_MAX_BATCH_SIZE, src1_ncols - col_offset);
        const char * y_ptr = src1_ddq_i + col_offset * stride_y;
        float * dst_ptr = dst_dd_i + col_offset * stride_dst;

        switch (src0->type) {
            case GGML_TYPE_Q4_0:
                if (use_reorder) {
                    for (int c = 0; c < ncols_this; ++c) {
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, y_ptr + c * stride_y, dst_ptr + c * stride_dst, ne00, row_diff, stream);
                    }
                } else {
                    CALL_BATCHED(mul_mat_vec_q4_0_q8_1_batched_sycl, ncols_this, src0_dd_i, y_ptr, dst_ptr, ne00, row_diff, stride_y, stride_dst, stream);
                }
                break;
            case GGML_TYPE_Q4_1:
                CALL_BATCHED(mul_mat_vec_q4_1_q8_1_batched_sycl, ncols_this, src0_dd_i, y_ptr, dst_ptr, ne00, row_diff, stride_y, stride_dst, stream);
                break;
            case GGML_TYPE_Q5_0:
                CALL_BATCHED(mul_mat_vec_q5_0_q8_1_batched_sycl, ncols_this, src0_dd_i, y_ptr, dst_ptr, ne00, row_diff, stride_y, stride_dst, stream);
                break;
            case GGML_TYPE_Q5_1:
                CALL_BATCHED(mul_mat_vec_q5_1_q8_1_batched_sycl, ncols_this, src0_dd_i, y_ptr, dst_ptr, ne00, row_diff, stride_y, stride_dst, stream);
                break;
            case GGML_TYPE_Q8_0:
                CALL_BATCHED(mul_mat_vec_q8_0_q8_1_batched_sycl, ncols_this, src0_dd_i, y_ptr, dst_ptr, ne00, row_diff, stride_y, stride_dst, stream);
                break;
            case GGML_TYPE_Q2_K:
                CALL_BATCHED(mul_mat_vec_q2_K_q8_1_batched_sycl, ncols_this, src0_dd_i, y_ptr, dst_ptr, ne00, row_diff, stride_y, stride_dst, stream);
                break;
            case GGML_TYPE_Q3_K:
                CALL_BATCHED(mul_mat_vec_q3_K_q8_1_batched_sycl, ncols_this, src0_dd_i, y_ptr, dst_ptr, ne00, row_diff, stride_y, stride_dst, stream);
                break;
            case GGML_TYPE_Q4_K:
                if (use_reorder) {
                    for (int c = 0; c < ncols_this; ++c) {
                        reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, y_ptr + c * stride_y, dst_ptr + c * stride_dst, ne00, row_diff, stream);
                    }
                } else {
                    CALL_BATCHED(mul_mat_vec_q4_K_q8_1_batched_sycl, ncols_this, src0_dd_i, y_ptr, dst_ptr, ne00, row_diff, stride_y, stride_dst, stream);
                }
                break;
            case GGML_TYPE_Q5_K:
                CALL_BATCHED(mul_mat_vec_q5_K_q8_1_batched_sycl, ncols_this, src0_dd_i, y_ptr, dst_ptr, ne00, row_diff, stride_y, stride_dst, stream);
                break;
            case GGML_TYPE_Q6_K:
                if (use_reorder) {
                    for (int c = 0; c < ncols_this; ++c) {
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, y_ptr + c * stride_y, dst_ptr + c * stride_dst, ne00, row_diff, stream);
                    }
                } else {
                    CALL_BATCHED(mul_mat_vec_q6_K_q8_1_batched_sycl, ncols_this, src0_dd_i, y_ptr, dst_ptr, ne00, row_diff, stride_y, stride_dst, stream);
                }
                break;
            case GGML_TYPE_IQ1_S:
                for (int c = 0; c < ncols_this; ++c) {
                    mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, y_ptr + c * stride_y, dst_ptr + c * stride_dst, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ1_M:
                for (int c = 0; c < ncols_this; ++c) {
                    mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, y_ptr + c * stride_y, dst_ptr + c * stride_dst, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ2_XXS:
                for (int c = 0; c < ncols_this; ++c) {
                    mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, y_ptr + c * stride_y, dst_ptr + c * stride_dst, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ2_XS:
                for (int c = 0; c < ncols_this; ++c) {
                    mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, y_ptr + c * stride_y, dst_ptr + c * stride_dst, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ2_S:
                for (int c = 0; c < ncols_this; ++c) {
                    mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, y_ptr + c * stride_y, dst_ptr + c * stride_dst, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ3_XXS:
                for (int c = 0; c < ncols_this; ++c) {
                    mul_mat_vec_iq3_xxs_q8_1_sycl(src0_dd_i, y_ptr + c * stride_y, dst_ptr + c * stride_dst, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ3_S:
                for (int c = 0; c < ncols_this; ++c) {
                    mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, y_ptr + c * stride_y, dst_ptr + c * stride_dst, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ4_NL:
                for (int c = 0; c < ncols_this; ++c) {
                    mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, y_ptr + c * stride_y, dst_ptr + c * stride_dst, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ4_XS:
                for (int c = 0; c < ncols_this; ++c) {
                    mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, y_ptr + c * stride_y, dst_ptr + c * stride_dst, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_MXFP4:
                CALL_BATCHED(mul_mat_vec_mxfp4_q8_1_batched_sycl, ncols_this, src0_dd_i, y_ptr, dst_ptr, ne00, row_diff, stride_y, stride_dst, stream);
                break;
            default:
                GGML_ABORT("fatal error");
        }
        col_offset += ncols_this;
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}
        col_offset += ncols_this;
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}
                break;
            case GGML_TYPE_Q4_1:
                mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q5_0:
                mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q5_1:
                mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q8_0:
                mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q2_K:
                mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q3_K:
                mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q4_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                    reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_K:
                mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q6_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                    reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_XXS:
                mul_mat_vec_iq3_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_S:
                mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_XS:
                mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_MXFP4:
                mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            default:
                GGML_ABORT("fatal error");
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}
