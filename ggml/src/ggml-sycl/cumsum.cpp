#include "cumsum.hpp"

// ---------------------------------------------------------------------------
// SYCL port of ggml-cuda/cumsum.cu (manual fallback kernel).
//
// Inclusive prefix sum (cumulative sum) along the innermost dimension (ne[0])
// of a tensor. One work-group handles one row; the work-group walks the row in
// tiles, carrying an accumulated sum across tiles.
//
// Algorithm per tile:
//   1. Each work-item loads num_unroll consecutive elements and computes a
//      local sequential prefix sum over them.
//   2. The per-work-item totals undergo an inclusive warp (sub-group) scan
//      using Hillis–Steele style shuffle-up.
//   3. Warp totals are gathered and scanned by warp 0 to produce per-warp
//      exclusive offsets.
//   4. Each work-item combines its local prefix sums with the warp offset
//      and the cross-tile carry, then writes results to global memory.
//   5. The tile total is added to the carry for the next tile.
//
// This mirrors the CUDA kernel 1:1, replacing __shfl_up_sync with
// sycl::shift_group_right and __syncthreads with group_barrier.
// ---------------------------------------------------------------------------

static constexpr int SYCL_CUMSUM_BLOCK_SIZE = 256;

// Warp-level inclusive prefix sum (Hillis–Steele)
// Uses sycl::shift_group_right which shifts values toward higher lane IDs,
// equivalent to CUDA's __shfl_up_sync.
template <int width = WARP_SIZE>
static inline float warp_prefix_inclusive_sum_sycl(float x, sycl::sub_group sg) {
    const int lane_id = sg.get_local_linear_id();
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const float t = sycl::shift_group_right(sg, x, offset);
        if (lane_id >= offset) {
            x += t;
        }
    }
    return x;
}

static void cumsum_f32_sycl(
        const float * src, float * dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t  nb1, const int64_t  nb2, const int64_t  nb3,
        sycl::queue & stream) {

    // Strides in elements (float)
    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t  s1 =  nb1 / sizeof(float);
    const int64_t  s2 =  nb2 / sizeof(float);
    const int64_t  s3 =  nb3 / sizeof(float);

    // Determine block size: enough warps to cover a row, capped at SYCL_CUMSUM_BLOCK_SIZE
    constexpr int warp_size = WARP_SIZE;
    const int num_warps_needed = static_cast<int>((ne00 + warp_size - 1) / warp_size);
    int block_size = num_warps_needed * warp_size;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);

    const int warps_per_block = block_size / warp_size;

    // Shared memory layout (in floats):
    //   s_vals[block_size]          - per-work-item scan values
    //   s_warp_sums[warps_per_block] - per-warp totals
    //   s_carry[1]                  - cross-tile carry
    //   s_chunk_total[1]            - total of current tile
    const int shmem_elems = block_size + warps_per_block + 2;

    // Grid: one work-group per row (ne01 * ne02 * ne03)
    const sycl::range<3> global_range(static_cast<size_t>(ne03),
                                       static_cast<size_t>(ne02),
                                       static_cast<size_t>(ne01) * block_size);
    const sycl::range<3> local_range(1, 1, static_cast<size_t>(block_size));

    stream.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> smem_acc(shmem_elems, h);

        // Capture all scalar/pointer values before the lambda — never capture ggml_tensor*
        const int blk = block_size;
        const int wpb = warps_per_block;
        const int64_t l_ne00 = ne00;
        const int64_t l_ne01 = ne01;
        const int64_t l_ne02 = ne02;
        const int64_t l_ne03 = ne03;
        const int64_t l_s01  = s01;
        const int64_t l_s02  = s02;
        const int64_t l_s03  = s03;
        const int64_t l_s1   = s1;
        const int64_t l_s2   = s2;
        const int64_t l_s3   = s3;

        h.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {

                const int tid = static_cast<int>(item.get_local_id(2));
                const int lane = tid % warp_size;
                const int warp = tid / warp_size;

                const int64_t i1 = static_cast<int64_t>(item.get_group(2));
                const int64_t i2 = static_cast<int64_t>(item.get_group(1));
                const int64_t i3 = static_cast<int64_t>(item.get_group(0));

                if (i1 >= l_ne01 || i2 >= l_ne02 || i3 >= l_ne03) {
                    return;
                }

                float * smem        = smem_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                float * s_vals      = smem;
                float * s_warp_sums = smem + blk;
                float * s_carry     = smem + blk + wpb;
                float * s_chunk_total = s_carry + 1;

                const float * src_row = src + i1 * l_s01 + i2 * l_s02 + i3 * l_s03;
                float       * dst_row = dst + i1 * l_s1  + i2 * l_s2  + i3 * l_s3;

                // Initialize carry
                if (tid == 0) {
                    *s_carry = 0.0f;
                }
                sycl::group_barrier(item.get_group());

                sycl::sub_group sg = item.get_sub_group();

                constexpr int num_unroll = 4;
                float temp[num_unroll];

                for (int64_t i = 0; i < l_ne00; i += num_unroll * blk) {
                    int64_t idx = i + tid * num_unroll;

                    // Thread-local sequential scan
                    temp[0] = (idx < l_ne00) ? src_row[idx] : 0.0f;
                    for (int j = 1; j < num_unroll; j++) {
                        temp[j] = temp[j - 1];
                        if (idx + j < l_ne00) {
                            temp[j] += src_row[idx + j];
                        }
                    }

                    // Last element is the sum of all values assigned to this work-item
                    float val = (idx < l_ne00) ? temp[num_unroll - 1] : 0.0f;

                    // Warp inclusive prefix scan
                    val = warp_prefix_inclusive_sum_sycl(val, sg);
                    s_vals[tid] = val;

                    if (lane == warp_size - 1) {
                        s_warp_sums[warp] = val;
                    }
                    sycl::group_barrier(item.get_group());

                    // Exclusive scan of warp sums (warp 0 only)
                    if (warp == 0) {
                        float w = (tid < wpb) ? s_warp_sums[tid] : 0.0f;
                        float inc = warp_prefix_inclusive_sum_sycl(w, sg);
                        if (tid < wpb) {
                            s_warp_sums[tid] = inc - w;   // exclusive prefix sum
                        }
                        if (tid == wpb - 1) {
                            *s_chunk_total = inc;          // total of this chunk
                        }
                    }
                    sycl::group_barrier(item.get_group());

                    // Write back results
                    float carry = *s_carry;
                    float final_val_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[num_unroll - 1];

                    for (int j = 0; j < num_unroll; j++) {
                        if (idx + j < l_ne00) {
                            dst_row[idx + j] = temp[j] + final_val_offset;
                        }
                    }

                    sycl::group_barrier(item.get_group());

                    // Update carry for next chunk
                    if (tid == 0) {
                        *s_carry += *s_chunk_total;
                    }
                }
            });
    });
}

// ---------------------------------------------------------------------------
// Entry point — called from ggml_sycl_compute_forward
// ---------------------------------------------------------------------------
void ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    // Extract all tensor fields before calling the kernel (never pass ggml_tensor* to device)
    const float * src0_d = (const float *) src0->data;
    float       * dst_d  = (float *) dst->data;

    sycl::queue & stream = *(ctx.stream());

    cumsum_f32_sycl(
        src0_d, dst_d,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src0->nb[1], src0->nb[2], src0->nb[3],
        dst->nb[1],  dst->nb[2],  dst->nb[3],
        stream
    );
}
