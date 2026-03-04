#include "softmax.hpp"
#include <cstdint>
#include <utility>
#include <cmath>

// opt_task_014: Enable vec4 vectorized loads/stores for memory-bound kernels
// Target: Intel Arc Pro B60 (Xe2), 456 GB/s memory bandwidth
// vec4 loads can improve memory bandwidth utilization by 20-40%
#ifndef GGML_SYCL_VEC4
#define GGML_SYCL_VEC4 1
#endif

template <typename T> static inline float t2f32(T val) {
    return (float) val;
}

template <> float inline t2f32<sycl::half>(sycl::half val) {
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

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we supress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <bool use_shared, int ncols_template, int block_size_template, typename T>
static void soft_max_f32(const float *         x,
                         const T *             mask,
                         const float *         sinks,
                         float *               dst,
                         const soft_max_params p,
                         uint8_t *             dpct_local) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    const int block_size = block_size_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_size_template;
    const int tid = item_ct1.get_local_id(2);

    const int64_t i03 = item_ct1.get_group(0);
    const int64_t i02 = item_ct1.get_group(1);
    const int64_t i01 = item_ct1.get_group(2);

    //TODO: noncontigous inputs/outputs
    const int rowx = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                         item_ct1.get_group_range(1);

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.ne12;
    const int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / sizeof(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    // opt_task_030: SLM used only for vals cache when use_shared=true.
    // Inter-warp reduction buffer (buf_iw) eliminated by using sycl::reduce_over_group.
    // shared memory buffer to cache values between iterations:
    float *vals = use_shared ? (float *) dpct_local : dst;
    float max_val = sinks ? sinks[i02] : -INFINITY;

    // opt_task_031/037: vec4 vectorized loads/stores for the three column loops.
    // Each thread processes 4 consecutive float elements per iteration,
    // improving memory bandwidth utilization on Xe2 (Intel Arc Pro B60).
    //
    // opt_task_037: Only use vec4 when ncols >= block_size * 4, meaning every
    // thread gets at least one full vec4 iteration. For small ncols (<=block_size*4),
    // vec4 leaves most threads idle (only ncols/4 threads participate) which is worse
    // than the scalar path where all block_size threads contribute.
    // Regression data: [64,64] was -28.6%, [1024,1024] was -7.5%, [256,256] was -2%.
#if GGML_SYCL_VEC4
    // Compile-time decision for template-specialized paths:
    // Use vec4 only when ncols_template >= block_size_template * 4
    // (i.e., every thread has at least one full vec4 iteration).
    // For the dynamic path (ncols_template == 0), decide at runtime.
    constexpr bool use_vec4_template =
        (ncols_template != 0) && (block_size_template != 0) &&
        (ncols_template >= block_size_template * 4);
    // Runtime decision for dynamic path
    const bool use_vec4 = (ncols_template == 0) ? (ncols >= block_size * 4) : use_vec4_template;
#endif

#if GGML_SYCL_VEC4
    if (use_vec4) {
        const int ncols_vec4 = (ncols / 4) * 4;

        // Loop 1: scale + mask + max (vec4 path)
#pragma unroll
        for (int col = tid * 4; col < ncols_vec4; col += block_size * 4) {
            sycl::vec<float, 4> x_v = *reinterpret_cast<const sycl::vec<float, 4>*>(&x[col]);
            sycl::vec<float, 4> val_v;
            if (mask) {
                val_v[0] = x_v[0]*p.scale + slope*t2f32(mask[col]);
                val_v[1] = x_v[1]*p.scale + slope*t2f32(mask[col+1]);
                val_v[2] = x_v[2]*p.scale + slope*t2f32(mask[col+2]);
                val_v[3] = x_v[3]*p.scale + slope*t2f32(mask[col+3]);
            } else {
                val_v = x_v * p.scale;
            }
            *reinterpret_cast<sycl::vec<float, 4>*>(&vals[col]) = val_v;
            max_val = sycl::max(max_val, sycl::max(sycl::max(val_v[0], val_v[1]), sycl::max(val_v[2], val_v[3])));
        }
        // Scalar tail for non-vec4-aligned remainder (only when ncols_template == 0)
        if (ncols_template == 0) {
            for (int col = ncols_vec4 + tid; col < ncols; col += block_size) {
                const float val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);
                vals[col] = val;
                max_val = sycl::max(max_val, val);
            }
        }
    } else {
#endif
#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
#if GGML_SYCL_VEC4
    }
#endif

    // opt_task_030: Use sycl::reduce_over_group for work-group max reduction.
    // Replaces manual 2-barrier SLM reduction (warp shuffle + buf_iw + 2 barriers).
    // On Xe2 (Intel Arc Pro B60), this lets the compiler/runtime pick the optimal
    // reduction tree, avoiding explicit SLM traffic and barrier stalls.
    max_val = sycl::reduce_over_group(item_ct1.get_group(), max_val, sycl::maximum<float>());
    float tmp = 0.0f; // partial sum

    // Loop 2: exp + sum (vec4 path when beneficial)
#if GGML_SYCL_VEC4
    if (use_vec4) {
        const int ncols_vec4 = (ncols / 4) * 4;
#pragma unroll
        for (int col = tid * 4; col < ncols_vec4; col += block_size * 4) {
            sycl::vec<float, 4> v = *reinterpret_cast<const sycl::vec<float, 4>*>(&vals[col]);
            sycl::vec<float, 4> exp_v;
            exp_v[0] = sycl::native::exp(v[0] - max_val);
            exp_v[1] = sycl::native::exp(v[1] - max_val);
            exp_v[2] = sycl::native::exp(v[2] - max_val);
            exp_v[3] = sycl::native::exp(v[3] - max_val);
            tmp += exp_v[0] + exp_v[1] + exp_v[2] + exp_v[3];
            *reinterpret_cast<sycl::vec<float, 4>*>(&vals[col]) = exp_v;
        }
        // Scalar tail for non-vec4-aligned remainder (only when ncols_template == 0)
        if (ncols_template == 0) {
            for (int col = ncols_vec4 + tid; col < ncols; col += block_size) {
                const float val = sycl::native::exp(vals[col] - max_val);
                tmp += val;
                vals[col] = val;
            }
        }
    } else {
#endif
#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = sycl::native::exp(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }
#if GGML_SYCL_VEC4
    }
#endif

    // opt_task_030: Use sycl::reduce_over_group for work-group sum reduction.
    // Replaces manual 3-barrier SLM reduction (warp shuffle + buf_iw zero + store + read + 3 barriers).
    // Combined with the max reduction above, this eliminates all 5 manual barriers
    // and the buf_iw SLM allocation, letting the SYCL runtime use the optimal
    // reduction strategy for Xe2 sub-group width (SIMD16).
    tmp = sycl::reduce_over_group(item_ct1.get_group(), tmp, sycl::plus<float>());
    if (sinks) {
        tmp += sycl::native::exp(sinks[i02] - max_val);
    }
    const float inv_sum = 1.0f / tmp;

    // Loop 3: normalize (vec4 path when beneficial)
#if GGML_SYCL_VEC4
    if (use_vec4) {
        const int ncols_vec4 = (ncols / 4) * 4;
#pragma unroll
        for (int col = tid * 4; col < ncols_vec4; col += block_size * 4) {
            sycl::vec<float, 4> v = *reinterpret_cast<const sycl::vec<float, 4>*>(&vals[col]);
            *reinterpret_cast<sycl::vec<float, 4>*>(&dst[col]) = v * inv_sum;
        }
        // Scalar tail for non-vec4-aligned remainder (only when ncols_template == 0)
        if (ncols_template == 0) {
            for (int col = ncols_vec4 + tid; col < ncols; col += block_size) {
                dst[col] = vals[col] * inv_sum;
            }
        }
    } else {
#endif
#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        dst[col] = vals[col] * inv_sum;
    }
#if GGML_SYCL_VEC4
    }
#endif
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

// opt_task_035: Softmax backward with multi-sub-group work-group (256 threads).
// Uses sycl::reduce_over_group instead of warp_reduce_sum for cross-sub-group reduction.
// 256 threads = 16 sub-groups of SIMD16, greatly improving occupancy over the
// previous single-warp (16 thread) work-group which used only 1/8 of EU capacity.
static void soft_max_back_f32(const float *grad, const float *dstf, float *dst,
                              const int ncols, const float scale) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid      = item_ct1.get_local_id(2);
    const int block_size = item_ct1.get_local_range(2);
    const int rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    float dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

#if GGML_SYCL_VEC4
    // Process 4 elements at a time with vec4 loads
    // Each thread processes consecutive vec4 elements: thread i handles elements [i*4, i*4+1, i*4+2, i*4+3]
    const int vec4_col_end = (ncols / 4) * 4;
    for (int col = tid * 4; col < vec4_col_end; col += block_size * 4) {
        sycl::vec<float, 4> grad_v = *reinterpret_cast<const sycl::vec<float, 4>*>(&grad[col]);
        sycl::vec<float, 4> dstf_v = *reinterpret_cast<const sycl::vec<float, 4>*>(&dstf[col]);
        dgf_dot += dstf_v[0] * grad_v[0] + dstf_v[1] * grad_v[1] + dstf_v[2] * grad_v[2] + dstf_v[3] * grad_v[3];
    }
    // Handle remaining elements (0-3)
    for (int col = vec4_col_end + tid; col < ncols; col += block_size) {
        dgf_dot += dstf[col]*grad[col];
    }
#else
    for (int col = tid; col < ncols; col += block_size) {
        dgf_dot += dstf[col]*grad[col];
    }
#endif

    // opt_task_035: Use reduce_over_group for multi-sub-group reduction
    // Replaces warp_reduce_sum which only worked within a single sub-group.
    dgf_dot = sycl::reduce_over_group(item_ct1.get_group(), dgf_dot, sycl::plus<float>());

#if GGML_SYCL_VEC4
    // Process 4 elements at a time with vec4 loads/stores
    for (int col = tid * 4; col < vec4_col_end; col += block_size * 4) {
        sycl::vec<float, 4> grad_v = *reinterpret_cast<const sycl::vec<float, 4>*>(&grad[col]);
        sycl::vec<float, 4> dstf_v = *reinterpret_cast<const sycl::vec<float, 4>*>(&dstf[col]);
        sycl::vec<float, 4> dst_v = scale * (grad_v - dgf_dot) * dstf_v;
        *reinterpret_cast<sycl::vec<float, 4>*>(&dst[col]) = dst_v;
    }
    // Handle remaining elements (0-3)
    for (int col = vec4_col_end + tid; col < ncols; col += block_size) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
#else
    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
#endif
}

template <int... Ns, typename T>
static void launch_soft_max_kernels(const float *           x,
                                    const T *               mask,
                                    const float *           sinks,
                                    float *                 dst,
                                    const soft_max_params & p,
                                    sycl::queue*         stream,
                                    dpct::dim3              block_dims,
                                    dpct::dim3              block_nums,
                                    size_t                  nbytes_shared)
{
    auto launch_kernel = [=](auto I) -> bool {
        constexpr int ncols = decltype(I)::value;
        // opt_task_037: Choose block size to minimize reduce_over_group overhead.
        //
        // reduce_over_group cost scales with sub-group count: each sub-group does a
        // local reduce, then cross-sub-group coordination via SLM + barriers. For
        // small ncols, this coordination dominates the total kernel time.
        //
        // Strategy:
        // - ncols <= WARP_SIZE (32): block = ncols (1-2 sub-groups, reduce is ~free)
        // - ncols <= 4*WARP_SIZE (64): block = WARP_SIZE (1 sub-group, each thread
        //   processes ncols/WARP_SIZE elements; reduces cross-sub-group sync to zero)
        // - ncols > 4*WARP_SIZE && ncols <= 256: block = ncols (reasonable sub-group
        //   count, and each thread processes 1 element — no loop overhead)
        // - ncols > 256: block = 256 (cap at 16 sub-groups to avoid excessive
        //   reduce_over_group overhead; each thread iterates ncols/256 times)
        //
        // Benchmarked regressions this fixes (vs pre-optimization baseline):
        //   [64,64,20,1]:     was -28.6% with block=64 (4 sub-groups)
        //   [1024,1024,10,1]: was  -7.5% with block=256 (16 sub-groups)
        //   [77,4096,5,1]:    was  -6.2% (fallback path, fixed by capping nth below)
        constexpr int block = (ncols <= WARP_SIZE ? ncols :
                               ncols <= 4 * WARP_SIZE ? WARP_SIZE :
                               ncols <= 256 ? ncols : 256);
        if (p.ncols == ncols) {
            // opt_task_037: Use the compile-time 'block' for the nd_range work-group
            // size, NOT the passed block_dims. This ensures the actual number of
            // launched threads matches block_size_template inside the kernel.
            // Without this, threads with tid >= block would access out-of-bounds
            // columns in the template-specialized (no bounds-check) loops.
            const dpct::dim3 opt_block(block, 1, 1);
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * opt_block, opt_block),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        WARP_SIZE)]] {
                        soft_max_f32<true, ncols, block>(
                            x, mask, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_UNUSED(item_ct1);
                    });
            });
            return true;
        }
        return false;
    };

    // unary fold over launch_kernel
    if ((launch_kernel(std::integral_constant<int, Ns>{}) || ...)) {
        return;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<true, 0, 0>(
                        x, mask, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
    });
}

template <typename T>
static void soft_max_f32_sycl(const float *x, const T *mask,
                              const float *sinks, float *dst,
                              const soft_max_params &params,
                              sycl::queue* stream, int device) {
    int nth = WARP_SIZE;
    int max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    const int64_t ncols_x = params.ncols;

    // opt_task_037: Minimize sub-group count in the work-group to reduce
    // reduce_over_group overhead (which scales with sub-group count).
    // For the template-specialized (shared) path, launch_soft_max_kernels
    // computes its own optimal block size per ncols; nth here is only used
    // for the fallback (ncols_template==0) and the non-shared path.
    while (nth < ncols_x && nth < max_block_size) nth *= 2;
    if (nth > max_block_size) nth = max_block_size;
    // For the shared-memory fallback path (ncols doesn't match any template),
    // cap at 256 to avoid excessive sub-group coordination overhead.
    // 256 = 16 sub-groups of SIMD16, which is the sweet spot on Xe2.
    const int nth_shared_fallback = (nth > 256) ? 256 : nth;

    const dpct::dim3 block_dims(nth, 1, 1);
    const dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    // opt_task_030: SLM only needed for vals cache (no buf_iw after reduce_over_group).
    const size_t nbytes_shared =
        GGML_PAD(ncols_x, WARP_SIZE) * sizeof(float);

    const int id       = get_current_device_id();
    const size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    if (nbytes_shared <= smpbo && ncols_x <= max_block_size) {
        // opt_task_037: block_dims here is only used for the fallback path
        // (when ncols doesn't match any template). Template-specialized paths
        // compute their own optimal block size inside launch_soft_max_kernels.
        const dpct::dim3 block_dims_fb(nth_shared_fallback, 1, 1);
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, mask, sinks, dst, params, stream, block_dims_fb, block_nums,
            nbytes_shared);
    } else {
        // opt_task_030: No SLM needed for non-shared path — buf_iw eliminated,
        // vals uses dst directly, and reduce_over_group manages its own scratch.
        const size_t nbytes_shared_low = 0;

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_low), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    soft_max_f32<false, 0, 0>(
                        x, mask, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
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
                                   sycl::queue* stream) {
    // opt_task_035: Increased from WARP_SIZE (16) to 256 for better occupancy.
    // 256 threads = 16 sub-groups of SIMD16, using full EU thread capacity.
    const dpct::dim3 block_dims(256, 1, 1);
    const dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

void ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *) src0->data;
    const void  * src1_d = src1 ? (const void *) src1->data : nullptr;
    const void  * src2_d = src2 ? (const void *) src2->data : nullptr;
    float       *  dst_d = (float *) dst->data;

    sycl::queue* stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
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

    if (use_f16) {
        soft_max_f32_sycl(src0_d, (const sycl::half *)src1_d,
                          (const float *)src2_d, dst_d, params, stream,
                          ctx.device);
    } else {
        soft_max_f32_sycl(src0_d, (const float *)src1_d, (const float *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

void ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0 = dst->src[0]; // grad
    const ggml_tensor * src1 = dst->src[1]; // forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    sycl::queue* stream = ctx.stream();

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
