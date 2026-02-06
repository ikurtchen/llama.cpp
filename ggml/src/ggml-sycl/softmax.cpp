#include "softmax.hpp"
#include <cstdint>
#include <utility>
#include <cmath>


template <typename T> static __dpct_inline__ float t2f32(T val) {
    return (float) val;
}

template <> float __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  return sycl::vec<sycl::half, 1>(val)
      .convert<float, sycl::rounding_mode::automatic>()[0];
}

struct soft_max_params {

    int64_t nheads;
    uint32_t n_head_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    int64_t ne12;
    int64_t ne13;
    float scale;
    float max_bias;
    float m0;
    float m1;
};

// Block reduction for max using shared memory
template <int block_size>
static inline float block_reduce_max(float val, float * shared_mem, const sycl::nd_item<3> & item_ct1) {
    const int tid = item_ct1.get_local_id(2);
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // First, reduce within warp
    val = warp_reduce_max(val, item_ct1);

    // Write reduced warp value to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = val;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Final reduction in first warp
    if (warp_id == 0) {
        val = (tid < (block_size / WARP_SIZE)) ? shared_mem[lane_id] : -INFINITY;
        val = warp_reduce_max(val, item_ct1);
    }

    // Broadcast the result to all threads
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (warp_id == 0 && lane_id == 0) {
        shared_mem[0] = val;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    return shared_mem[0];
}

// Block reduction for sum using shared memory
template <int block_size>
static inline float block_reduce_sum(float val, float * shared_mem, const sycl::nd_item<3> & item_ct1) {
    const int tid = item_ct1.get_local_id(2);
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // First, reduce within warp
    val = warp_reduce_sum(val, item_ct1);

    // Write reduced warp value to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = val;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Final reduction in first warp
    if (warp_id == 0) {
        val = (tid < (block_size / WARP_SIZE)) ? shared_mem[lane_id] : 0.0f;
        val = warp_reduce_sum(val, item_ct1);
    }

    // Broadcast the result to all threads
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (warp_id == 0 && lane_id == 0) {
        shared_mem[0] = val;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    return shared_mem[0];
}

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we supress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

// Softmax kernel: computes softmax with optional mask and ALiBi bias
// use_shared: whether to use shared memory for caching intermediate values
// ncols_template: compile-time ncols (0 means runtime)
// block_size_template: compile-time block size (0 means runtime)
template <bool use_shared, int ncols_template, int block_size_template, typename T>
static void soft_max_f32_kernel(
        const float * x,
        const T * mask,
        const float * sinks,
        float * dst,
        const soft_max_params p,
        const sycl::nd_item<3> & item_ct1,
        float * buf_shared) {

    const int ncols = ncols_template == 0 ? p.ncols : ncols_template;

    const int tid = item_ct1.get_local_id(2);

    const int64_t i03 = item_ct1.get_group(0);
    const int64_t i02 = item_ct1.get_group(1);
    const int64_t i01 = item_ct1.get_group(2);

    const int rowx = item_ct1.get_group(2) + item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) * item_ct1.get_group_range(1);

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.ne12;
    const int64_t i13 = i03 % p.ne13;

    const float * x_row = x + int64_t(rowx) * ncols;
    const T * mask_row = mask ? mask + (i11 * p.nb11 + i12 * p.nb12 + i13 * p.nb13) / sizeof(T) : nullptr;
    float * dst_row = dst + int64_t(rowx) * ncols;

    const int block_size = block_size_template == 0 ? item_ct1.get_local_range(2) : block_size_template;

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    // Shared memory layout: buf_iw for inter-warp communication, vals for caching
    float * buf_iw = buf_shared;
    float * vals = use_shared ? buf_iw + WARP_SIZE : dst_row;

    // Pass 1: find max value
    float max_val = sinks ? sinks[i02] : -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = x_row[col] * p.scale + (mask_row ? slope * t2f32(mask_row[col]) : 0.0f);

        vals[col] = val;
        max_val = sycl::fmax(max_val, val);
    }

    // Block reduction for max
    max_val = block_reduce_max<block_size_template == 0 ? SYCL_SOFT_MAX_BLOCK_SIZE : block_size_template>(max_val, buf_iw, item_ct1);

    // Pass 2: compute exp(x - max) and sum
    float tmp = 0.0f;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = sycl::exp(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    // Block reduction for sum
    tmp = block_reduce_sum<block_size_template == 0 ? SYCL_SOFT_MAX_BLOCK_SIZE : block_size_template>(tmp, buf_iw, item_ct1);

    if (sinks) {
        tmp += sycl::exp(sinks[i02] - max_val);
    }

    const float inv_sum = 1.0f / tmp;

    // Pass 3: normalize by sum
#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        dst_row[col] = vals[col] * inv_sum;
    }
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

static void soft_max_back_f32(const float *grad, const float *dstf, float *dst,
                              const int ncols, const float scale) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid      = item_ct1.get_local_id(2);
    const int rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    float dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum(dgf_dot);

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template <typename T>
static void soft_max_f32_sycl(const float *x, const T *mask,
                              const float *sinks, float *dst,
                              const soft_max_params &params,
                              dpct::queue_ptr stream, int device) {
    int nth = WARP_SIZE;
    const int64_t ncols_x = params.ncols;

    // Scale up thread count to next power of 2, capped at SYCL_SOFT_MAX_BLOCK_SIZE
    while (nth < ncols_x && nth < SYCL_SOFT_MAX_BLOCK_SIZE) {
        nth *= 2;
    }

    const sycl::range<3> block_dims(1, 1, nth);
    const sycl::range<3> block_nums(params.ne03, params.ne02, params.ne01);

    // Shared memory: WARP_SIZE for inter-warp communication + ncols for caching values
    const size_t nbytes_shared = (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE) * sizeof(float);

    // Check if we can use shared memory for caching intermediate values
    // For now, we use a simplified approach: always use shared memory if it fits
    const size_t max_shared_mem = 65536; // Conservative limit for Intel GPUs

    if (nbytes_shared <= max_shared_mem) {
        // Launch with shared memory caching (use_shared = true)
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> buf_shared_acc(sycl::range<1>(GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    float * buf_shared = buf_shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                    soft_max_f32_kernel<true, 0, 0>(x, mask, sinks, dst, params, item_ct1, buf_shared);
                });
        });
    } else {
        // Launch without shared memory caching (use_shared = false)
        // Values are stored directly in destination array
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> buf_shared_acc(sycl::range<1>(WARP_SIZE), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    float * buf_shared = buf_shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                    soft_max_f32_kernel<false, 0, 0>(x, mask, sinks, dst, params, item_ct1, buf_shared);
                });
        });
    }
}

static void soft_max_back_f32_sycl(const float *   grad,
                                   const float *   dstf,
                                   float *         dst,
                                   const int       ncols,
                                   const int       nrows,
                                   const float     scale,
                                   dpct::queue_ptr stream) {
    const dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    const dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

void ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1]; // mask (optional)
    const ggml_tensor * src2 = dst->src[2]; // sinks (optional)

    const float * src0_d = (const float *) src0->data;
    const void  * src1_d = src1 ? (const void *) src1->data : nullptr;
    const void  * src2_d = src2 ? (const void *) src2->data : nullptr;
    float       * dst_d  = (float *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional, can be F16 or F32
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    const int64_t ne00 = src0->ne[0];

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    const int64_t nb11 = src1 ? src1->nb[1] : 1;
    const int64_t nb12 = src1 ? src1->nb[2] : 1;
    const int64_t nb13 = src1 ? src1->nb[3] : 1;

    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    const uint32_t n_head      = src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_head_log2 = n_head_log2;
    params.ncols = ne00;
    params.nrows_x = nrows_x;
    params.nrows_y = nrows_y;
    params.ne00 = src0->ne[0];
    params.ne01 = src0->ne[1];
    params.ne02 = src0->ne[2];
    params.ne03 = src0->ne[3];
    params.nb11 = nb11;
    params.nb12 = nb12;
    params.nb13 = nb13;
    params.ne12 = ne12;
    params.ne13 = ne13;
    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;

    const int device = ctx.device;

    if (use_f16) {
        soft_max_f32_sycl(src0_d, (const sycl::half *) src1_d, (const float *) src2_d, dst_d, params, stream, device);
    } else {
        soft_max_f32_sycl(src0_d, (const float *) src1_d, (const float *) src2_d, dst_d, params, stream, device);
    }
}

void ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0 = dst->src[0]; // grad
    const ggml_tensor * src1 = dst->src[1]; // forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
