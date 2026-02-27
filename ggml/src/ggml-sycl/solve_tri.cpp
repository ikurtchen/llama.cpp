#include "solve_tri.hpp"

#include "common.hpp"

#define MAX_N_FAST 64
#define MAX_K_FAST 32

// ======================
// Batch pointer setup kernel (for oneMKL fallback path)
// ======================
static void get_batch_pointers(const float *  A,
                               float *        X,
                               const float ** A_ptrs,
                               float **       X_ptrs,
                               int64_t        ne02,
                               int64_t        total_batches,
                               size_t         s02,
                               size_t         s03,
                               size_t         s2,
                               size_t         s3,
                               const sycl::nd_item<1> & item) {
    const int idx = item.get_global_id(0);
    if (idx >= total_batches) {
        return;
    }

    const int64_t i3 = idx / ne02;
    const int64_t i2 = idx % ne02;

    A_ptrs[idx] = A + i3 * s03 + i2 * s02;
    X_ptrs[idx] = X + i3 * s3  + i2 * s2;
}

// ======================
// Fast Kernel (n <= 64, k <= 32) - Warp-based parallel reduction
//
// The CUDA version uses WARP_SIZE=32 and splits into 2 halves (low/high).
// On SYCL/Intel, WARP_SIZE can be 16 or 32. We generalize to
// NUM_SECTIONS = MAX_N_FAST / WARP_SIZE sections so the kernel works
// for any sub-group width.
// ======================
#define NUM_SECTIONS (MAX_N_FAST / WARP_SIZE)

template <int n_template, int k_template>
static void solve_tri_f32_fast(const float * __restrict__ A,
                               const float * __restrict__ B,
                               float * __restrict__ X,
                               const uint32_t ne02_mp,
                               const uint32_t ne02_L,
                               const uint32_t ne02_d,
                               const size_t nb02,
                               const size_t nb03,
                               const size_t nb12,
                               const size_t nb13,
                               const size_t nb2,
                               const size_t nb3,
                               const int    n_arg,
                               const int    k_arg,
                               const sycl::nd_item<3> & item_ct1,
                               float * sA) {
    const int n = n_template == 0 ? n_arg : n_template;
    const int k = k_template == 0 ? k_arg : k_template;

    const int batch_idx = item_ct1.get_group(0);   // blockIdx.x
    const int col_idx   = item_ct1.get_local_id(1); // threadIdx.y
    const int lane      = item_ct1.get_local_id(2); // threadIdx.x

    if (col_idx >= k) {
        return;
    }

    const sycl::uint3 ne02 = sycl::uint3(ne02_mp, ne02_L, ne02_d);
    const sycl::uint2 i02_i03 = fast_div_modulo(batch_idx, ne02);
    const int64_t i02 = i02_i03.y();
    const int64_t i03 = i02_i03.x();

    const float * const A_batch = (const float *)(A + i02 * nb02 + i03 * nb03);
    const float * const B_batch = (const float *)(B + i02 * nb12 + i03 * nb13);
    float *             X_batch = (float *)(X + i02 * nb2 + i03 * nb3);

    // Load A into shared memory cooperatively
    const int offset = item_ct1.get_local_id(2) + item_ct1.get_local_id(1) * item_ct1.get_local_range(2);

    // opt_task_015: Enable full unroll for compile-time-constant inner loops
    // See: hw_spec_b60.md, optimization_guide
    #pragma unroll
    for (int i = 0; i < n * n; i += k * WARP_SIZE) {
        const int i0 = i + offset;
        if (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    sycl::group_barrier(item_ct1.get_group());

    // Load B values for this column — one per section
    float x[NUM_SECTIONS];
    // opt_task_015: Enable full unroll for compile-time-constant inner loops
    // See: hw_spec_b60.md, optimization_guide
    #pragma unroll
    for (int s = 0; s < NUM_SECTIONS; ++s) {
        const int row = s * WARP_SIZE + lane;
        x[s] = (row < n) ? B_batch[row * k + col_idx] : 0.0f;
    }

    // Number of sections actually used
    const int n_sections = (n + WARP_SIZE - 1) / WARP_SIZE;

    // Forward substitution — process one section at a time
    for (int sec = 0; sec < n_sections; ++sec) {
        const int row_start = sec * WARP_SIZE;
        const int row_end   = (row_start + WARP_SIZE < n) ? row_start + WARP_SIZE : n;

        // opt_task_015: Enable full unroll for compile-time-constant inner loops
        // See: hw_spec_b60.md, optimization_guide
        #pragma unroll
        for (int row = row_start; row < row_end; ++row) {
            float sum = 0.0f;

            // Accumulate contributions from all fully-solved prior sections
            // opt_task_015: Enable full unroll for compile-time-constant inner loops
            // See: hw_spec_b60.md, optimization_guide
            #pragma unroll
            for (int ps = 0; ps < sec; ++ps) {
                sum += sA[row * n + ps * WARP_SIZE + lane] * x[ps];
            }

            // Accumulate contributions from current section (partial — only lanes < row within section)
            const int lane_row = sec * WARP_SIZE + lane;
            if (lane_row < row) {
                sum += sA[row * n + lane_row] * x[sec];
            }

            sum = warp_reduce_sum(sum, item_ct1);

            if (lane == row - row_start) {
                x[sec] = (x[sec] - sum) / sA[row * n + row];
            }
        }
    }

    // Write results
    // opt_task_015: Enable full unroll for compile-time-constant inner loops
    // See: hw_spec_b60.md, optimization_guide
    #pragma unroll
    for (int s = 0; s < NUM_SECTIONS; ++s) {
        const int row = s * WARP_SIZE + lane;
        if (row < n) {
            X_batch[row * k + col_idx] = x[s];
        }
    }
}

#undef NUM_SECTIONS

// ======================
// Fast kernel launcher with template dispatch
// ======================
static void solve_tri_f32_sycl(const float * A,
                                const float * B,
                                float *       X,
                                int           n,
                                int           k,
                                int64_t       ne02,
                                int64_t       ne03,
                                size_t        nb02,
                                size_t        nb03,
                                size_t        nb12,
                                size_t        nb13,
                                size_t        nb2,
                                size_t        nb3,
                                dpct::queue_ptr stream) {
    const sycl::uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);

    // SYCL nd_range<3>: dim0=batch, dim1=col_idx(k), dim2=lane(WARP_SIZE)
    const sycl::range<3> block_dims(1, k, WARP_SIZE);
    const sycl::range<3> grid_dims(ne02 * ne03, k, WARP_SIZE);

    // Macro to reduce repetition in template dispatch
    const uint32_t ne02_mp = ne02_fd.x();
    const uint32_t ne02_L  = ne02_fd.y();
    const uint32_t ne02_d  = ne02_fd.z();

    #define LAUNCH_SOLVE_TRI_FAST(N_TPL, K_TPL, N_ARG, K_ARG)                           \
        stream->submit([&](sycl::handler & cgh) {                                        \
            sycl::local_accessor<float, 1> sA_acc(MAX_N_FAST * MAX_N_FAST, cgh);         \
            cgh.parallel_for(                                                             \
                sycl::nd_range<3>(grid_dims, block_dims),                                 \
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] { \
                    solve_tri_f32_fast<N_TPL, K_TPL>(                                     \
                        A, B, X, ne02_mp, ne02_L, ne02_d,                                 \
                        nb02, nb03, nb12, nb13, nb2, nb3,                                 \
                        N_ARG, K_ARG, item_ct1,                                           \
                        sA_acc.get_multi_ptr<sycl::access::decorated::no>().get());       \
                });                                                                       \
        })

    if (n == 64) {
        switch (k) {
            case 32: LAUNCH_SOLVE_TRI_FAST(64, 32, 0, 0); break;
            case 16: LAUNCH_SOLVE_TRI_FAST(64, 16, 0, 0); break;
            case 14: LAUNCH_SOLVE_TRI_FAST(64, 14, 0, 0); break;
            case 12: LAUNCH_SOLVE_TRI_FAST(64, 12, 0, 0); break;
            case 10: LAUNCH_SOLVE_TRI_FAST(64, 10, 0, 0); break;
            case  8: LAUNCH_SOLVE_TRI_FAST(64,  8, 0, 0); break;
            case  6: LAUNCH_SOLVE_TRI_FAST(64,  6, 0, 0); break;
            case  4: LAUNCH_SOLVE_TRI_FAST(64,  4, 0, 0); break;
            case  2: LAUNCH_SOLVE_TRI_FAST(64,  2, 0, 0); break;
            case  1: LAUNCH_SOLVE_TRI_FAST(64,  1, 0, 0); break;
            default: LAUNCH_SOLVE_TRI_FAST( 0,  0, n, k); break;
        }
    } else {
        LAUNCH_SOLVE_TRI_FAST(0, 0, n, k);
    }

    #undef LAUNCH_SOLVE_TRI_FAST
}

// ======================
// oneMKL trsm_batch path for large matrices (n > 64 or k > 32)
// ======================
static void solve_tri_f32_onemkl(ggml_backend_sycl_context & ctx,
                                  const float *               A,
                                  const float *               B,
                                  float *                     X,
                                  int                         n,
                                  int                         k,
                                  int64_t                     ne02,
                                  int64_t                     ne03,
                                  size_t                      s02,
                                  size_t                      s03,
                                  size_t                      s12,
                                  size_t                      s13,
                                  size_t                      s2,
                                  size_t                      s3,
                                  dpct::queue_ptr             stream) {
    const float   alpha         = 1.0f;
    const int64_t total_batches = ne02 * ne03;
    if (total_batches == 0) {
        return;
    }

    // Bulk copy B -> X (contiguous tensors)
    if (X != B) {
        const int64_t total_elements_BX = (int64_t)n * k * total_batches;
        stream->memcpy(X, B, total_elements_BX * sizeof(float));
    }

    // Allocate batch pointer arrays from pool
    ggml_sycl_pool_alloc<const float *> A_ptrs_alloc(ctx.pool(), total_batches);
    ggml_sycl_pool_alloc<float *>       X_ptrs_alloc(ctx.pool(), total_batches);

    const float ** A_ptrs_dev = A_ptrs_alloc.get();
    float **       X_ptrs_dev = X_ptrs_alloc.get();

    // Launch kernel to compute per-batch pointers
    {
        const int block_size = 256;
        const int grid_size  = (total_batches + block_size - 1) / block_size;
        stream->parallel_for(
            sycl::nd_range<1>(grid_size * block_size, block_size),
            [=](sycl::nd_item<1> item) {
                get_batch_pointers(A, X, A_ptrs_dev, X_ptrs_dev, ne02,
                                   total_batches, s02, s03, s2, s3, item);
            });
    }

    // oneMKL trsm_batch: solve X * A = B  (right-side, upper triangular)
    // Equivalent to cuBLAS: cublasStrsmBatched(handle, SIDE_RIGHT, FILL_UPPER, OP_N, DIAG_NON_UNIT, k, n, ...)
    auto side  = oneapi::math::side::right;
    auto uplo  = oneapi::math::uplo::upper;
    auto trans  = oneapi::math::transpose::nontrans;
    auto diag  = oneapi::math::diag::nonunit;
    int64_t m_64          = k;
    int64_t n_64          = n;
    int64_t lda_64        = n;
    int64_t ldb_64        = k;
    int64_t group_size_64 = total_batches;

    try {
        oneapi::math::blas::column_major::trsm_batch(
            get_onemath_backend(*stream),
            &side, &uplo, &trans, &diag,
            &m_64, &n_64, &alpha,
            (const float **)A_ptrs_dev, &lda_64,
            X_ptrs_dev, &ldb_64,
            1,              // group_count
            &group_size_64, // group_sizes array
            {});            // dependencies
    } catch (sycl::exception const & exc) {
        std::cerr << exc.what() << std::endl;
        GGML_ASSERT(false);
    }

    GGML_UNUSED(s12);
    GGML_UNUSED(s13);
}

// ======================
// Public entry point
// ======================
void ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * src0 = dst->src[0];  // A (n x n, lower triangular)
    const ggml_tensor * src1 = dst->src[1];  // B (n x k)

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int64_t n    = src0->ne[0];
    const int64_t k    = src1->ne[0];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    if (n <= MAX_N_FAST && k <= MAX_K_FAST) {
        solve_tri_f32_sycl(
            (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
            n, k, ne02, ne03,
            src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
            src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
            dst->nb[2]  / sizeof(float), dst->nb[3]  / sizeof(float),
            stream);
    } else {
        solve_tri_f32_onemkl(
            ctx,
            (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
            n, k, ne02, ne03,
            src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
            src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
            dst->nb[2]  / sizeof(float), dst->nb[3]  / sizeof(float),
            stream);
    }
}
