#include "cumsum.hpp"

#include "common.hpp"

#include <algorithm>

static void cumsum_f32(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t  s1, const int64_t  s2, const int64_t  s3,
        const sycl::nd_item<3> & item_ct1,
        float * s_vals, float * s_warp_sums, float * s_carry) {

    const int tid        = item_ct1.get_local_id(2);
    const int block_size = item_ct1.get_local_range(2);
    constexpr int warp_size_local = WARP_SIZE;
    const int lane             = tid % warp_size_local;
    const int warp             = tid / warp_size_local;
    const int warps_per_block  = block_size / warp_size_local;

    // s_carry layout: s_carry[0] = carry, s_carry[1] = chunk_total
    float * s_chunk_total = s_carry + 1;

    // Initialize carry
    if (tid == 0) {
        *s_carry = 0.0f;
    }
    sycl::group_barrier(item_ct1.get_group());

    const int64_t i3 = item_ct1.get_group(0);
    const int64_t i2 = item_ct1.get_group(1);
    const int64_t i1 = item_ct1.get_group(2);
    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    const float * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    float       * dst_row = dst + i1 * s1  + i2 * s2  + i3 * s3;

    constexpr int num_unroll = 4;
    float temp[num_unroll];

    const auto sg = item_ct1.get_sub_group();

    for (int64_t i = 0; i < ne00; i += num_unroll * block_size) {
        int64_t idx = i + tid * num_unroll;

        // Thread-local sequential scan
        temp[0] = (idx < ne00) ? src_row[idx] : 0.0f;
        for (int j = 1; j < num_unroll; j++) {
            temp[j] = temp[j - 1];
            if (idx + j < ne00) {
                temp[j] += src_row[idx + j];
            }
        }

        // Last element is sum of all values assigned to this thread
        float val = (idx < ne00) ? temp[num_unroll - 1] : 0.0f;

        // Warp inclusive prefix scan using SYCL sub-group algorithm
        val = sycl::inclusive_scan_over_group(sg, val, sycl::plus<float>());
        s_vals[tid] = val;

        if (lane == warp_size_local - 1) {
            s_warp_sums[warp] = val;
        }
        sycl::group_barrier(item_ct1.get_group());

        // Exclusive scan of warp sums (warp 0 only)
        if (warp == 0) {
            float w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            float inc = sycl::inclusive_scan_over_group(sg, w, sycl::plus<float>());
            if (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;  // exclusive sum
            }
            if (tid == warps_per_block - 1) {
                *s_chunk_total = inc;  // total sum of this chunk
            }
        }
        sycl::group_barrier(item_ct1.get_group());

        // Write back results
        float carry = *s_carry;
        // Calculate sum offset for this thread
        float final_val_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[num_unroll - 1];

        for (int j = 0; j < num_unroll; j++) {
            if (idx + j < ne00) {
                dst_row[idx + j] = temp[j] + final_val_offset;
            }
        }

        sycl::group_barrier(item_ct1.get_group());

        // Update carry for next chunk
        if (tid == 0) {
            *s_carry += *s_chunk_total;
        }
    }
}

void ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t nb1 = dst->nb[1];
    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    const float * src0_d = (const float *) src0->data;
    float       * dst_d  = (float       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    // Compute block size (same logic as CUDA)
    const int num_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    int block_size = num_warps * WARP_SIZE;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);

    const int warps_per_block = block_size / WARP_SIZE;

    // Shared memory sizes
    const int shmem_vals      = block_size;
    const int shmem_warp_sums = warps_per_block;
    const int shmem_carry     = 2;  // carry + chunk_total

    // Convert strides from bytes to elements
    const size_t type_size = sizeof(float);
    const int64_t s01 = nb01 / type_size;
    const int64_t s02 = nb02 / type_size;
    const int64_t s03 = nb03 / type_size;
    const int64_t  s1 = nb1  / type_size;
    const int64_t  s2 = nb2  / type_size;
    const int64_t  s3 = nb3  / type_size;

    // Grid: one work-group per row (ne03 x ne02 x ne01 to match CUDA grid(ne01, ne02, ne03))
    const sycl::range<3> block_dims(1, 1, block_size);
    const sycl::range<3> grid_dims(ne03, ne02, ne01);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> s_vals_acc(sycl::range<1>(shmem_vals), cgh);
        sycl::local_accessor<float, 1> s_warp_sums_acc(sycl::range<1>(shmem_warp_sums), cgh);
        sycl::local_accessor<float, 1> s_carry_acc(sycl::range<1>(shmem_carry), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cumsum_f32(
                    src0_d, dst_d,
                    ne00, ne01, ne02, ne03,
                    s01, s02, s03,
                    s1, s2, s3,
                    item_ct1,
                    s_vals_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    s_warp_sums_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    s_carry_acc.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}
