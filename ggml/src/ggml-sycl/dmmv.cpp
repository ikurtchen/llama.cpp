#include "convert.hpp"
#include "dmmv.hpp"
#include "dequantize.hpp"
#include "presets.hpp"

// ============================================================================
// Optimization: task_017 - Increase work-group size for DMMV kernels
// Target: Intel Arc Pro B60 (Xe2 architecture, SIMD16, 20 Xe-cores)
// 
// Original: WARP_SIZE=16 threads per work-group (single sub-group)
// Optimized: DMMV_BLOCK_SIZE=128 threads (8 sub-groups of 16 threads each)
//
// Key changes:
// 1. Increased work-group size from 16 to 128 threads
// 2. Added two-level reduction:
//    - Sub-group (warp) level reduction using dpct::permute_sub_group_by_xor
//    - Work-group level reduction using SLM (local_accessor)
// 3. Each sub-group leader writes partial result to SLM
// 4. First sub-group performs final reduction across all sub-groups
//
// Expected improvement: 4-8x throughput on Intel Xe2 due to better
// utilization of XVEs (only 1 hardware thread per XVE was active before)
// ============================================================================

#ifndef DMMV_BLOCK_SIZE
#define DMMV_BLOCK_SIZE 128
#endif

#define DMMV_K_QUANT_REDUCTION(tmp, item_ct1, slm_partial, sg_id, tid_in_sg, dst, row) \
    do { \
        _Pragma("unroll") \
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) { \
            tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask); \
        } \
        constexpr int num_sub_groups = QK_WARP_SIZE / WARP_SIZE; \
        if (tid_in_sg == 0) { \
            slm_partial[sg_id] = tmp; \
        } \
        item_ct1.barrier(sycl::access::fence_space::local_space); \
        if (sg_id == 0) { \
            float final_sum = (tid_in_sg < num_sub_groups) ? slm_partial[tid_in_sg] : 0.0f; \
            for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) { \
                final_sum += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), final_sum, mask); \
            } \
            if (tid_in_sg == 0) { \
                dst[row] = final_sum; \
            } \
        } \
    } while(0)

static void convert_f16(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const sycl::half *x = (const sycl::half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

static void convert_f32(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const float * x = (const float *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

#ifdef GGML_SYCL_HAS_BF16
static void convert_bf16(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const sycl::ext::oneapi::bfloat16 *x = (const sycl::ext::oneapi::bfloat16 *)vx;

    v.x() = (float)x[ib + iqs + 0];
    v.y() = (float)x[ib + iqs + 1];
}
#endif

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, int block_size = DMMV_BLOCK_SIZE>
static void dequantize_mul_mat_vec(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows,
                                   const sycl::nd_item<3> &item_ct1, float * __restrict__ slm_partial) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int row = item_ct1.get_group(2);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);
    const int sg_id = tid / WARP_SIZE;
    const int tid_in_sg = tid % WARP_SIZE;
    const int num_sub_groups = block_size / WARP_SIZE;

    const int iter_stride = 2*GGML_SYCL_DMMV_X * (block_size / WARP_SIZE);
    const int vals_per_iter = iter_stride / block_size;
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f};
#else
    float tmp = 0.0f;
#endif // GGML_SYCL_F16

    for (int i = tid * vals_per_iter; i < ncols; i += iter_stride) {
        const int col = i;
        if (col >= ncols) break;
        const int ib = (row*ncols + col)/qk;
        const int iqs = (col%qk)/qr;
        const int iybs = col - col%qk;

#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            const int curr_col = col + j;
            if (curr_col >= ncols) break;
            const int curr_ib = (row*ncols + curr_col)/qk;
            const int curr_iqs = (curr_col%qk)/qr;
            const int curr_iybs = curr_col - curr_col%qk;

            dfloat2 v;
            dequantize_kernel(vx, curr_ib, curr_iqs, v);

#ifdef GGML_SYCL_F16
            dfloat2 t1{y[curr_iybs + curr_iqs + 0],
                        y[curr_iybs + curr_iqs + y_offset]};
            tmp += v * t1;
#else
            tmp += v.x() * y[curr_iybs + curr_iqs + 0];
            tmp += v.y() * y[curr_iybs + curr_iqs + y_offset];
#endif
        }
    }

    // Sub-group (warp) level reduction
    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    // Two-level work-group reduction using SLM
    if constexpr (block_size > WARP_SIZE) {
        // Each sub-group leader writes partial result to SLM
        if (tid_in_sg == 0) {
#ifdef GGML_SYCL_F16
            slm_partial[sg_id] = tmp.x() + tmp.y();
#else
            slm_partial[sg_id] = tmp;
#endif
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // First sub-group performs final reduction
        if (sg_id == 0) {
            float final_sum = (tid_in_sg < num_sub_groups) ? slm_partial[tid_in_sg] : 0.0f;
            for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
                final_sum += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), final_sum, mask);
            }
            if (tid_in_sg == 0) {
                dst[row] = final_sum;
            }
        }
    } else {
        // Single sub-group: original behavior
        if (tid == 0) {
#ifdef GGML_SYCL_F16
            dst[row] = tmp.x() + tmp.y();
#else
            dst[row] = tmp;
#endif
        }
    }
}

template <int qk, int qr, dequantize_kernel_t_reorder dequantize_kernel_reorder, int block_size = DMMV_BLOCK_SIZE>
static void dequantize_mul_mat_vec_reorder(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows,
                                   const sycl::nd_item<3> &item_ct1, float * __restrict__ slm_partial) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int row = item_ct1.get_group(2);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);
    const int sg_id = tid / WARP_SIZE;
    const int tid_in_sg = tid % WARP_SIZE;
    const int num_sub_groups = block_size / WARP_SIZE;

    const int ncols_left = ncols % (QK4_0*block_size);
    const int ncols_align = ncols - ncols_left;
    const int iter_stride = 8*2*GGML_SYCL_DMMV_X * (block_size / WARP_SIZE);
    const int vals_per_iter = iter_stride / block_size;
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f};
#else
    float tmp = 0.0f;
#endif
    const char *d_ptr = (const char*)vx+ncols*nrows/2;
    int i=0;
    for (i = tid * vals_per_iter; i < ncols_align; i += iter_stride) {
        const int col = i;
        if (col >= ncols_align) break;
        const int ib = (row*ncols + col)/qk;
        const int iqs = (col%qk)/qr;
        const int iybs = col - col%qk;

#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            const int curr_col = col + j;
            if (curr_col >= ncols_align) break;
            const int curr_ib = (row*ncols + curr_col)/qk;
            const int curr_iqs = (curr_col%qk)/qr;
            const int curr_iybs = curr_col - curr_col%qk;

            dfloat2 v;
            dequantize_kernel_reorder((const void *)d_ptr, curr_ib, (const void *)vx, curr_ib * QK4_0 / 2 + curr_iqs, v);

#ifdef GGML_SYCL_F16
            dfloat2 t1{y[curr_iybs + curr_iqs + 0],
                        y[curr_iybs + curr_iqs + y_offset]};
            tmp += v * t1;
#else
            tmp += v.x() * y[curr_iybs + curr_iqs + 0];
            tmp += v.y() * y[curr_iybs + curr_iqs + y_offset];
#endif
        }
    }

    for (; i < ncols; i += iter_stride) {
        const int col = tid * vals_per_iter + (i - ncols_align);
        if (col >= ncols_left) break;
        const int actual_col = ncols_align + col;
        const int ib = (row*ncols + actual_col)/qk;
        const int iqs = (actual_col%qk)/qr;
        const int iybs = actual_col - actual_col%qk;

#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            const int curr_col = actual_col + j;
            if (curr_col >= ncols) break;
            const int curr_ib = (row*ncols + curr_col)/qk;
            const int curr_iqs = (curr_col%qk)/qr;
            const int curr_iybs = curr_col - curr_col%qk;

            dfloat2 v;
            dequantize_kernel_reorder((const void *)d_ptr, curr_ib, (const void *)vx, curr_ib * QK4_0 / 2 + curr_iqs, v);

#ifdef GGML_SYCL_F16
            dfloat2 t1{y[curr_iybs + curr_iqs + 0],
                        y[curr_iybs + curr_iqs + y_offset]};
            tmp += v * t1;
#else
            tmp += v.x() * y[curr_iybs + curr_iqs + 0];
            tmp += v.y() * y[curr_iybs + curr_iqs + y_offset];
#endif
        }
    }

    // Sub-group (warp) level reduction
    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    // Two-level work-group reduction using SLM
    if constexpr (block_size > WARP_SIZE) {
        if (tid_in_sg == 0) {
#ifdef GGML_SYCL_F16
            slm_partial[sg_id] = tmp.x() + tmp.y();
#else
            slm_partial[sg_id] = tmp;
#endif
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        if (sg_id == 0) {
            float final_sum = (tid_in_sg < num_sub_groups) ? slm_partial[tid_in_sg] : 0.0f;
            for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
                final_sum += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), final_sum, mask);
            }
            if (tid_in_sg == 0) {
                dst[row] = final_sum;
            }
        }
    } else {
        if (tid == 0) {
#ifdef GGML_SYCL_F16
            dst[row] = tmp.x() + tmp.y();
#else
            dst[row] = tmp;
#endif
        }
    }
}

static void convert_mul_mat_vec_f16_sycl(const void *vx, const dfloat *y,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = nrows;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, 1, DMMV_BLOCK_SIZE);
    const int num_sub_groups = DMMV_BLOCK_SIZE / WARP_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> slm_partial_acc_ct1(num_sub_groups, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    dequantize_mul_mat_vec<1, 1, convert_f16, DMMV_BLOCK_SIZE>(
                        vx, y, dst, ncols, nrows, item_ct1,
                        slm_partial_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

#ifdef GGML_SYCL_HAS_BF16
static void convert_mul_mat_vec_bf16_sycl(const void *vx, const dfloat *y,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = nrows;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, 1, DMMV_BLOCK_SIZE);
    const int num_sub_groups = DMMV_BLOCK_SIZE / WARP_SIZE;
    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_partial_acc_ct1(num_sub_groups, cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<1, 1, convert_bf16, DMMV_BLOCK_SIZE>(
                    vx, y, dst, ncols, nrows, item_ct1,
                    slm_partial_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}
#endif

// ============================================================================
// Optimization: task_018 - Use SIMD16 sub-groups for K-quant DMMV kernels
// Target: Intel Arc Pro B60 (Xe2 architecture, SIMD16 native width)
// 
// Original: QK_WARP_SIZE=32 forcing SIMD32, halving occupancy on Xe2
// Optimized: Use WARP_SIZE=16 sub-groups with two-level reduction
//
// Key changes:
// 1. Changed reqd_sub_group_size from QK_WARP_SIZE (32) to WARP_SIZE (16)
// 2. Added two-level reduction: sub-group level + SLM work-group level
// 3. Each sub-group leader writes partial result to SLM
// 4. First sub-group performs final reduction across all sub-groups
//
// Expected improvement: 2x occupancy on Intel Xe2 due to native SIMD16 usage
// ============================================================================

/*
DPCT1110:4: The total declared local variable size in device function
dequantize_mul_mat_vec_q2_k exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void dequantize_mul_mat_vec_q2_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1,
                                        float * __restrict__ slm_partial) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int tid = item_ct1.get_local_id(2);
    const int sg_id = tid / WARP_SIZE;
    const int tid_in_sg = tid % WARP_SIZE;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q2_K * x = (const block_q2_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const int tid_k =
        tid / K_QUANTS_PER_ITERATION; // 0...31 or 0...15
    const int ix =
        tid % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int im = tid_k/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid_k - step*im;                        // 0...15 or 0...7

    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2
    const int q_offset = 32*im + l0;
    const int s_offset = 8*im;
    const int y_offset = 128*im + l0;

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + y_offset;
        const uint8_t * q = x[i].qs + q_offset;

        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        const uint32_t * a = (const uint32_t *)(x[i].scales + s_offset);
        aux[0] = a[0] & 0x0f0f0f0f;
        aux[1] = a[1] & 0x0f0f0f0f;
        aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
        aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                  + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                  + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                  + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                  + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                  + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                  + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                  +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
            sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                  + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];

        }
        tmp += dall * sum1 - dmin * sum2;

    }
#else
    const int tid_k = tid /
                    (2 * K_QUANTS_PER_ITERATION); // 0...15 or 0...7
    const int ix = tid %
                   (2 * K_QUANTS_PER_ITERATION); // 0....1 or 0...3
    const int offset = tid_k * K_QUANTS_PER_ITERATION;

    uint32_t uaux[2];
    const uint8_t * d = (const uint8_t *)uaux;


    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint32_t * s = (const uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        const sycl::float2 dall =
            x[i].dm.convert<float, sycl::rounding_mode::automatic>();

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x() * sum1 - dall.y() * sum2;
    }

#endif

    DMMV_K_QUANT_REDUCTION(tmp, item_ct1, slm_partial, sg_id, tid_in_sg, dst, row);
}

/*
DPCT1110:5: The total declared local variable size in device function
dequantize_mul_mat_vec_q3_k exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void dequantize_mul_mat_vec_q3_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1,
                                        float * __restrict__ slm_partial) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int tid = item_ct1.get_local_id(2);
    const int sg_id = tid / WARP_SIZE;
    const int tid_in_sg = tid % WARP_SIZE;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q3_K * x = (const block_q3_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid_k =
        tid / K_QUANTS_PER_ITERATION; // 0...31 or 0...16
    const int ix =
        tid % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    const int step = 16/K_QUANTS_PER_ITERATION;
    const int im = tid_k/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid_k - step*im;                        // 0....15 or 0...7

    const uint8_t m = 1 << (4*im);

    const int l0 = n*in;                                 // 0...15 or 0...14 in steps of 2
    const int q_offset =  32*im + l0;
    const int y_offset = 128*im + l0;

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    const uint16_t s_shift = 4*im;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * q = x[i].qs + q_offset;
        const uint8_t * h = x[i].hmask + l0;

        const uint16_t * a = (const uint16_t *)x[i].scales;
        utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
        utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
        utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
        utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

        const float d = x[i].d;

        float sum = 0;
        for (int l = 0; l < n; ++l) {
            sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                 + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                 + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                 + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
            sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                 + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                 + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
        }
        tmp += d * sum;

    }
#else

    const int tid_k = tid/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    const int ix  = tid%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    const int offset = tid_k * K_QUANTS_PER_ITERATION;         // 0...15 or 0...14
    const int in = offset/8;                                 // 0 or 1
    const int im = offset%8;                                 // 0...7

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint8_t * s = x[i].scales;

        const float dall = (float)x[i].d;

        float sum = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t hl = x[i].hmask[im+l] >> in;
            const uint8_t ql = q[l];
            sum += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += sum;
    }
#endif

    DMMV_K_QUANT_REDUCTION(tmp, item_ct1, slm_partial, sg_id, tid_in_sg, dst, row);
}

/*
DPCT1110:6: The total declared local variable size in device function
dequantize_mul_mat_vec_q4_k exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void dequantize_mul_mat_vec_q4_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1,
                                        float * __restrict__ slm_partial) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int tid = item_ct1.get_local_id(2);
    const int sg_id = tid / WARP_SIZE;
    const int tid_in_sg = tid % WARP_SIZE;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q4_K * x = (const block_q4_K *)vx + ib0;

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid_k =
        tid / K_QUANTS_PER_ITERATION; // 0...31 or 0...16
    const int ix =
        tid % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    const int il  = tid_k/step;                            // 0...3
    const int ir  = tid_k - step*il;                       // 0...7 or 0...3
    const int n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

#if K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    const uint8_t * q4 = (const uint8_t *)q32;
#else
    uint16_t q16[4];
    const uint8_t * q4 = (const uint8_t *)q16;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y1 = yy + i*QK_K + y_offset;
        const float   * y2 = y1 + 128;

        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#if K_QUANTS_PER_ITERATION == 2
        const uint32_t * q1 = (const uint32_t *)(x[i].qs + q_offset);
        const uint32_t * q2 = q1 + 16;

        q32[0] = q1[0] & 0x0f0f0f0f;
        q32[1] = q1[0] & 0xf0f0f0f0;
        q32[2] = q2[0] & 0x0f0f0f0f;
        q32[3] = q2[0] & 0xf0f0f0f0;

        sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 4; ++l) {
            s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
            s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                       s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
               dmin * smin;
#else
        const uint16_t * q1 = (const uint16_t *)(x[i].qs + q_offset);
        const uint16_t * q2 = q1 + 32;

        q16[0] = q1[0] & 0x0f0f;
        q16[1] = q1[0] & 0xf0f0;
        q16[2] = q2[0] & 0x0f0f;
        q16[3] = q2[0] & 0xf0f0;

        float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 2; ++l) {
            s.x += y1[l] * q4[l+0]; s.y += y1[l+32] * q4[l+2];
            s.z += y2[l] * q4[l+4]; s.w += y2[l+32] * q4[l+6];
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp += dall * (s.x * sc[0] + s.y * sc[1] * 1.f/16.f + s.z * sc[4] + s.w * sc[5] * 1.f/16.f) - dmin * smin;
#endif

    }
#else
    const int tid_k = tid/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = tid%(2*K_QUANTS_PER_ITERATION);

    const int step = tid_k * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const float   * y = yy + i*QK_K + step;
        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        const float d = (float)x[i].dm[0];
        const float m = (float)x[i].dm[1];
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += sum;
    }

#endif

    DMMV_K_QUANT_REDUCTION(tmp, item_ct1, slm_partial, sg_id, tid_in_sg, dst, row);
}

/*
DPCT1110:7: The total declared local variable size in device function
dequantize_mul_mat_vec_q5_k exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void dequantize_mul_mat_vec_q5_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols,
                                        const sycl::nd_item<3> &item_ct1,
                                        float * __restrict__ slm_partial) {

    const int row = item_ct1.get_group(2);
    const int tid = item_ct1.get_local_id(2);
    const int sg_id = tid / WARP_SIZE;
    const int tid_in_sg = tid % WARP_SIZE;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q5_K * x = (const block_q5_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid_k = tid / 2; // 0...15
    const int ix = tid % 2;

    const int il  = tid_k/4;     // 0...3
    const int ir  = tid_k - 4*il;// 0...3
    const int n   = 2;

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    const uint8_t hm1  = 1 << (2*im);
    const uint8_t hm2  = hm1 << 4;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint16_t q16[8];
    const uint8_t * q4 = (const uint8_t *)q16;

    for (int i = ix; i < num_blocks_per_row; i += 2) {

        const uint8_t * ql1 = x[i].qs + q_offset;
        const uint8_t * qh  = x[i].qh + l0;
        const float   * y1  = yy + i*QK_K + y_offset;
        const float   * y2  = y1 + 128;

        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

        sycl::float4 sum = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        const uint16_t * q1 = (const uint16_t *)ql1;
        const uint16_t * q2 = q1 + 32;
        q16[0] = q1[0] & 0x0f0f;
        q16[1] = q1[8] & 0x0f0f;
        q16[2] = (q1[0] >> 4) & 0x0f0f;
        q16[3] = (q1[8] >> 4) & 0x0f0f;
        q16[4] = q2[0] & 0x0f0f;
        q16[5] = q2[8] & 0x0f0f;
        q16[6] = (q2[0] >> 4) & 0x0f0f;
        q16[7] = (q2[8] >> 4) & 0x0f0f;
        for (int l = 0; l < n; ++l) {
            sum.x() +=
                y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
            sum.y() +=
                y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
            sum.z() +=
                y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
            sum.w() +=
                y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
            smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                  + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
        }
        tmp += dall * (sum.x() * sc[0] + sum.y() * sc[1] + sum.z() * sc[4] +
                       sum.w() * sc[5]) -
               dmin * smin;
    }

#else
    const int tid_k = tid/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = tid%(2*K_QUANTS_PER_ITERATION);
    const int step = tid_k * K_QUANTS_PER_ITERATION;
    const int im = step/8;
    const int in = step%8;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const int8_t  * s = x[i].scales;
        const float   * y = yy + i*QK_K + step;
        const float     d = x[i].d;
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            const uint8_t h = x[i].qh[in+j] >> im;
            sum += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += sum;
    }
#endif

    DMMV_K_QUANT_REDUCTION(tmp, item_ct1, slm_partial, sg_id, tid_in_sg, dst, row);
}

static void dequantize_mul_mat_vec_q6_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1, float * __restrict__ slm_partial) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int tid = item_ct1.get_local_id(2);
    const int sg_id = tid / WARP_SIZE;
    const int tid_in_sg = tid % WARP_SIZE;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q6_K * x = (const block_q6_K *)vx + ib0;

#if QK_K == 256

    const int tid_k =
        tid / K_QUANTS_PER_ITERATION; // 0...31 or 0...16
    const int ix =
        tid % K_QUANTS_PER_ITERATION; // 0 or 0, 1

    const int step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    const int im = tid_k/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid_k - step*im;                        // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    const int is = 0;
#else
    const int l0 = 4 * in;                               // 0, 4, 8, ..., 28
    const int is = in / 4;
#endif
    const int ql_offset = 64*im + l0;
    const int qh_offset = 32*im + l0;
    const int s_offset  =  8*im + is;
    const int y_offset = 128*im + l0;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * ql = x[i].ql + ql_offset;
        const uint8_t * qh = x[i].qh + qh_offset;
        const int8_t  * s  = x[i].scales + s_offset;

        const float d = x[i].d;

#if K_QUANTS_PER_ITERATION == 1
        float sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                  + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                  + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                  + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                  + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                  + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                  + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                  +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
        tmp += sum;
#else
        float sum = 0;
        for (int l = 0; l < 4; ++l) {
            sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                 + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                 + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                 + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
        tmp += sum;
#endif

    }

#else

    const int tid_k = tid/(2*K_QUANTS_PER_ITERATION);  // 0...7
    const int ix  = tid%(2*K_QUANTS_PER_ITERATION);  // 0...3

    const int step = tid_k * K_QUANTS_PER_ITERATION;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + step;
        const uint8_t * ql = x[i].ql + step;
        const uint8_t * qh = x[i].qh + step;
        const int8_t  * s  = x[i].scales;

        const float d = x[i+0].d;

        float sum = 0;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;

    }

#endif

    DMMV_K_QUANT_REDUCTION(tmp, item_ct1, slm_partial, sg_id, tid_in_sg, dst, row);
}

static void dequantize_mul_mat_vec_q4_0_sycl_reorder(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = nrows;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, 1, DMMV_BLOCK_SIZE);
    const int num_sub_groups = DMMV_BLOCK_SIZE / WARP_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> slm_partial_acc_ct1(num_sub_groups, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    dequantize_mul_mat_vec_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder, DMMV_BLOCK_SIZE>(
                        vx, y, dst, ncols, nrows, item_ct1,
                        slm_partial_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}


static void dequantize_mul_mat_vec_q4_0_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = nrows;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, 1, DMMV_BLOCK_SIZE);
    const int num_sub_groups = DMMV_BLOCK_SIZE / WARP_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> slm_partial_acc_ct1(num_sub_groups, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0, DMMV_BLOCK_SIZE>(
                        vx, y, dst, ncols, nrows, item_ct1,
                        slm_partial_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

static void dequantize_mul_mat_vec_q4_1_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = nrows;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, 1, DMMV_BLOCK_SIZE);
    const int num_sub_groups = DMMV_BLOCK_SIZE / WARP_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> slm_partial_acc_ct1(num_sub_groups, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1, DMMV_BLOCK_SIZE>(
                        vx, y, dst, ncols, nrows, item_ct1,
                        slm_partial_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

static void dequantize_mul_mat_vec_q5_0_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = nrows;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, 1, DMMV_BLOCK_SIZE);
    const int num_sub_groups = DMMV_BLOCK_SIZE / WARP_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> slm_partial_acc_ct1(num_sub_groups, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0, DMMV_BLOCK_SIZE>(
                        vx, y, dst, ncols, nrows, item_ct1,
                        slm_partial_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

static void dequantize_mul_mat_vec_q5_1_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = nrows;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, 1, DMMV_BLOCK_SIZE);
    const int num_sub_groups = DMMV_BLOCK_SIZE / WARP_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> slm_partial_acc_ct1(num_sub_groups, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1, DMMV_BLOCK_SIZE>(
                        vx, y, dst, ncols, nrows, item_ct1,
                        slm_partial_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

static void dequantize_mul_mat_vec_q8_0_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = nrows;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, 1, DMMV_BLOCK_SIZE);
    const int num_sub_groups = DMMV_BLOCK_SIZE / WARP_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> slm_partial_acc_ct1(num_sub_groups, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0, DMMV_BLOCK_SIZE>(
                        vx, y, dst, ncols, nrows, item_ct1,
                        slm_partial_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

static void dequantize_mul_mat_vec_q2_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, QK_WARP_SIZE);
    constexpr int num_sub_groups = QK_WARP_SIZE / WARP_SIZE;
    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_partial_acc_ct1(num_sub_groups, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q2_k(vx, y, dst, ncols, nrows, item_ct1,
                    slm_partial_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}

static void dequantize_mul_mat_vec_q3_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, QK_WARP_SIZE);
    constexpr int num_sub_groups = QK_WARP_SIZE / WARP_SIZE;
    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_partial_acc_ct1(num_sub_groups, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q3_k(vx, y, dst, ncols, nrows, item_ct1,
                    slm_partial_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}

static void dequantize_mul_mat_vec_q4_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, QK_WARP_SIZE);
    constexpr int num_sub_groups = QK_WARP_SIZE / WARP_SIZE;
    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_partial_acc_ct1(num_sub_groups, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q4_k(vx, y, dst, ncols, nrows, item_ct1,
                    slm_partial_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}

static void dequantize_mul_mat_vec_q5_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const sycl::range<3> block_dims(1, 1, QK_WARP_SIZE);
    constexpr int num_sub_groups = QK_WARP_SIZE / WARP_SIZE;
    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_partial_acc_ct1(num_sub_groups, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nrows) * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q5_k(vx, y, dst, ncols, item_ct1,
                    slm_partial_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}

static void dequantize_mul_mat_vec_q6_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, QK_WARP_SIZE);
    constexpr int num_sub_groups = QK_WARP_SIZE / WARP_SIZE;
    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm_partial_acc_ct1(num_sub_groups, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q6_k(vx, y, dst, ncols, nrows, item_ct1,
                    slm_partial_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}

void ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#ifdef GGML_SYCL_F16
    ggml_sycl_pool_alloc<sycl::half> src1_dfloat_a(ctx.pool());
    sycl::half *src1_dfloat = nullptr; // dfloat == half

    bool src1_convert_f16 =
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_F16;

    if (src1_convert_f16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");
        src1_dfloat = src1_dfloat_a.alloc(ne00);
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_ddf_i, src1_dfloat, ne00, stream);
    }
#else
    const dfloat * src1_dfloat = (const dfloat *) src1_ddf_i; // dfloat == float, no conversion
#endif // GGML_SYCL_F16

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            if ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q4_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            } else {
                dequantize_mul_mat_vec_q4_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_mul_mat_vec_q8_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_q2_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_q3_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q4_K:
            if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                // reorder is currently not supported for dmmv
                GGML_ABORT("Unimplemented dequantize case case for q4_k reorder");
            } else {
                dequantize_mul_mat_vec_q4_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_q5_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_q6_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_F16:
            convert_mul_mat_vec_f16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            convert_mul_mat_vec_bf16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
#endif
        default:
            printf("ggml_sycl_op_dequantize_mul_mat_vec unsupported GGML_TYPE %d\n", src0->type);
            GGML_ABORT("fatal error");
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
    GGML_UNUSED(ctx);
}
