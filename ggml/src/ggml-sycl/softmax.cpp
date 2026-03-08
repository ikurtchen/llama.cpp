#include "softmax.hpp"

#include <cstdint>
#include <cstring>
#include <cfloat>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

// Template to convert mask type to float
template <typename T>
static inline float t2f32(T val) {
    return (float)val;
}

// soft_max_f32 SYCL kernel
// Template on mask type T (float or sycl::half) and use_shared (whether vals are cached in local memory)
// ncols_template / block_size_template: compile-time specialization for ncols / block_size (0 = runtime)
template <typename T, bool use_shared, int ncols_template = 0, int block_size_template = 0>
static void soft_max_f32_sycl(
        const float * x, const T * mask, const float * sinks, float * dst,
        const int64_t ncols, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t ne12, const int64_t ne13,
        const float scale, const float max_bias,
        const uint32_t n_head_log2, const float m0, const float m1,
        sycl::queue & stream) {

    constexpr unsigned int block_size_const = block_size_template > 0 ? static_cast<unsigned int>(block_size_template) : 256u;
    const int block_size = static_cast<int>(block_size_const);
    const int64_t nrows = ne01 * ne02 * ne03;
    const int64_t padded_ncols = ((ncols + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    // local memory: WARP_SIZE floats for block_reduce inter-warp comm, plus optionally padded_ncols for value cache
    const int64_t shared_vals_count = use_shared ? (WARP_SIZE + padded_ncols) : WARP_SIZE;

    const sycl::range<1> global(nrows * block_size);
    const sycl::range<1> local(block_size);

    stream.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> shared_mem(shared_vals_count, h);

        h.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int ncols_eff = ncols_template > 0 ? ncols_template : static_cast<int>(ncols);
                const int tid = item.get_local_id(0);
                const int64_t rowx = item.get_group(0);

                // Decompose rowx into (i01, i02, i03)
                const int64_t i03 = rowx / (ne01 * ne02);
                const int64_t i02 = (rowx - i03 * ne01 * ne02) / ne01;
                const int64_t i01 = rowx - i03 * ne01 * ne02 - i02 * ne01;

                const float * x_row = x + rowx * ncols;
                float * dst_row = dst + rowx * ncols;

                // Mask addressing
                const int64_t i11 = i01;
                const int64_t i12 = i02 % ne12;
                const int64_t i13 = i03 % ne13;
                const T * mask_row = mask ? (mask + (i11 * nb11 + i12 * nb12 + i13 * nb13) / sizeof(T)) : nullptr;

                // ALiBi slope
                const float slope = get_alibi_slope(max_bias, i02, n_head_log2, m0, m1);

                // Pointers into local memory
                float * buf_iw = shared_mem.get_multi_ptr<sycl::access::decorated::no>().get_raw();
                float * vals = use_shared ? (buf_iw + WARP_SIZE) : dst_row;

                // Pass 1: load values, find max
                float max_val = sinks ? sinks[i02] : -INFINITY;

                #pragma unroll
                for (int col0 = 0; col0 < ncols_eff; col0 += block_size) {
                    const int col = col0 + tid;
                    if (ncols_template == 0 && col >= ncols_eff) {
                        break;
                    }

                    const float val = x_row[col] * scale + (mask_row ? slope * t2f32(mask_row[col]) : 0.0f);
                    vals[col] = val;
                    max_val = sycl::fmax(max_val, val);
                }

                // Block reduce MAX + broadcast
                max_val = block_reduce<block_reduce_method::MAX, block_size_const>(max_val, buf_iw, item);
                if (tid == 0) {
                    buf_iw[0] = max_val;
                }
                item.barrier(sycl::access::fence_space::local_space);
                max_val = buf_iw[0];

                // Pass 2: compute exp(val - max), sum
                float tmp = 0.0f;
                #pragma unroll
                for (int col0 = 0; col0 < ncols_eff; col0 += block_size) {
                    const int col = col0 + tid;
                    if (ncols_template == 0 && col >= ncols_eff) {
                        break;
                    }

                    const float val = sycl::exp(vals[col] - max_val);
                    tmp += val;
                    vals[col] = val;
                }

                // Block reduce SUM + broadcast
                tmp = block_reduce<block_reduce_method::SUM, block_size_const>(tmp, buf_iw, item);

                if (sinks) {
                    tmp += sycl::exp(sinks[i02] - max_val);
                }

                if (tid == 0) {
                    buf_iw[0] = tmp;
                }
                item.barrier(sycl::access::fence_space::local_space);
                tmp = buf_iw[0];

                const float inv_sum = 1.0f / tmp;

                // Pass 3: normalize
                #pragma unroll
                for (int col0 = 0; col0 < ncols_eff; col0 += block_size) {
                    const int col = col0 + tid;
                    if (ncols_template == 0 && col >= ncols_eff) {
                        break;
                    }

                    dst_row[col] = vals[col] * inv_sum;
                }
            }
        );
    });
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

// Helper template to dispatch ncols specializations
// Tries compile-time specializations for common ncols values, falls back to runtime
template <typename T>
static void soft_max_dispatch_sycl(
        const float * x, const T * mask, const float * sinks, float * dst,
        const int64_t ncols, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t ne12, const int64_t ne13,
        const float scale, const float max_bias,
        const uint32_t n_head_log2, const float m0, const float m1,
        const bool use_shared, sycl::queue & stream) {

    // Compile-time specializations: ncols -> block_size
    // block_size capped at WARP_SIZE * WARP_SIZE (= 256 on Xe2) because block_reduce
    // uses a single warp to reduce inter-warp partials, limiting to WARP_SIZE warps max.
    // use_shared=true for all specializations (ncols <= 4096 fits in SLM)
    constexpr int max_block_size = WARP_SIZE * WARP_SIZE; // 256 on Xe2
    #define SOFT_MAX_DISPATCH_CASE(NCOLS, BS) \
        if (ncols == NCOLS) { \
            soft_max_f32_sycl<T, true, NCOLS, BS>( \
                x, mask, sinks, dst, ncols, ne01, ne02, ne03, \
                nb11, nb12, nb13, ne12, ne13, \
                scale, max_bias, n_head_log2, m0, m1, stream); \
            return; \
        }

    if (use_shared) {
        SOFT_MAX_DISPATCH_CASE(32,   32)
        SOFT_MAX_DISPATCH_CASE(64,   64)
        SOFT_MAX_DISPATCH_CASE(128,  128)
        SOFT_MAX_DISPATCH_CASE(256,  256)
        SOFT_MAX_DISPATCH_CASE(512,  max_block_size)
        SOFT_MAX_DISPATCH_CASE(1024, max_block_size)
        SOFT_MAX_DISPATCH_CASE(2048, max_block_size)
        SOFT_MAX_DISPATCH_CASE(4096, max_block_size)
    }

    #undef SOFT_MAX_DISPATCH_CASE

    // Fallback: generic kernel with runtime ncols
    if (use_shared) {
        soft_max_f32_sycl<T, true, 0, 0>(
            x, mask, sinks, dst, ncols, ne01, ne02, ne03,
            nb11, nb12, nb13, ne12, ne13,
            scale, max_bias, n_head_log2, m0, m1, stream);
    } else {
        soft_max_f32_sycl<T, false, 0, 0>(
            x, mask, sinks, dst, ncols, ne01, ne02, ne03,
            nb11, nb12, nb13, ne12, ne13,
            scale, max_bias, n_head_log2, m0, m1, stream);
    }
}

// Host dispatch for soft_max
void ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1]; // mask (optional)
    const ggml_tensor * src2 = dst->src[2]; // sinks (optional)

    const float * src0_d = (const float *) src0->data;
    const void  * src1_d = src1 ? (const void *) src1->data : nullptr;
    const float * src2_d = src2 ? (const float *) src2->data : nullptr;
    float * dst_d = (float *) dst->data;

    sycl::queue & stream = *ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t nb11 = src1 ? src1->nb[1] : 1;
    const int64_t nb12 = src1 ? src1->nb[2] : 1;
    const int64_t nb13 = src1 ? src1->nb[3] : 1;
    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const uint32_t n_head = src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));
    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // Determine if row fits in local memory
    // We need (WARP_SIZE + padded_ncols) * sizeof(float) bytes
    const int64_t padded_ncols = ((ne00 + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    const int64_t shared_bytes_needed = (WARP_SIZE + padded_ncols) * sizeof(float);
    // Use conservative limit (48KB, matching typical SYCL local memory)
    const bool use_shared = (shared_bytes_needed <= 49152);

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    if (use_f16) {
        soft_max_dispatch_sycl<sycl::half>(src0_d, (const sycl::half *) src1_d, src2_d, dst_d,
            ne00, ne01, ne02, ne03, nb11, nb12, nb13, ne12, ne13,
            scale, max_bias, n_head_log2, m0, m1, use_shared, stream);
    } else {
        soft_max_dispatch_sycl<float>(src0_d, (const float *) src1_d, src2_d, dst_d,
            ne00, ne01, ne02, ne03, nb11, nb12, nb13, ne12, ne13,
            scale, max_bias, n_head_log2, m0, m1, use_shared, stream);
    }
}

// soft_max_back_f32 SYCL kernel
void ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // grad
    const ggml_tensor * src1 = dst->src[1]; // forward output

    const float * grad_d = (const float *) src0->data;
    const float * dstf_d = (const float *) src1->data;
    float * dst_d = (float *) dst->data;

    sycl::queue & stream = *ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    GGML_ASSERT(max_bias == 0.0f);

    const sycl::range<1> global(nrows * WARP_SIZE);
    const sycl::range<1> local(WARP_SIZE);

    stream.submit([&](sycl::handler & h) {
        h.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int tid = item.get_local_id(0);
                const int64_t rowx = item.get_group(0);

                const float * grad = grad_d + rowx * ncols;
                const float * dstf = dstf_d + rowx * ncols;
                float * out = dst_d + rowx * ncols;

                // Compute dot product of grad and forward output
                float dgf_dot = 0.0f;
                for (int col = tid; col < ncols; col += WARP_SIZE) {
                    dgf_dot += dstf[col] * grad[col];
                }
                dgf_dot = warp_reduce_sum(dgf_dot);

                // Compute output
                for (int col = tid; col < ncols; col += WARP_SIZE) {
                    out[col] = scale * (grad[col] - dgf_dot) * dstf[col];
                }
            }
        );
    });
}
