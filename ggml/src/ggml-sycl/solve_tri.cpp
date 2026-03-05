#include "solve_tri.hpp"

// ============================================================================
// Forward substitution kernel for lower-triangular solve: A * X = B
// where A is n×n lower triangular, B is n×k, X is n×k.
//
// Each work-group handles one (batch, column) pair.
// Threads within a work-group cooperatively compute the forward substitution
// for one column of X across all n rows.
//
// For n <= MAX_N_SHARED, the entire A matrix is loaded into shared memory.
// For larger n, A is read from global memory.
// ============================================================================

// Maximum n for which we load A into shared memory
// n=128 means 128*128*4 = 64KB of shared memory — fits in most Intel GPUs
#define MAX_N_SHARED 128

// --------------------------------------------------------------------------
// General forward-substitution kernel
// Works for any n. Uses shared memory for A when n <= MAX_N_SHARED.
//
// Launch: nd_range<1>, global = n_workgroups * block_size, local = block_size
// Work-group ID encodes (batch, col) pair.
// --------------------------------------------------------------------------
static void solve_tri_f32_kernel(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ X,
    const sycl::uint3    ne02_fd,
    const size_t         nb02,
    const size_t         nb03,
    const size_t         nb12,
    const size_t         nb13,
    const size_t         nb2,
    const size_t         nb3,
    const int            n,
    const int            k,
    const int            total_batches,
    const sycl::nd_item<1> & item,
    float * shared_mem)     // local (shared) memory
{
    const int wg_id   = item.get_group(0);
    const int tid     = item.get_local_id(0);
    const int block_sz = item.get_local_range(0);

    // Decode work-group ID into (batch_idx, col_idx)
    const int col_idx   = wg_id / total_batches;
    const int batch_idx = wg_id % total_batches;

    if (col_idx >= k) {
        return;
    }

    // Decode batch indices using fast division
    const sycl::uint2 i02_i03 = fast_div_modulo(batch_idx, ne02_fd);
    const int64_t i02 = i02_i03.y();
    const int64_t i03 = i02_i03.x();

    const float * A_batch = A + i03 * nb03 + i02 * nb02;
    const float * B_batch = B + i03 * nb13 + i02 * nb12;
    float *       X_batch = X + i03 * nb3  + i02 * nb2;

    // Shared memory layout:
    //   sA[n*n]  (only if n <= MAX_N_SHARED, else 0)
    //   sX[n]    (the column being solved)
    //   reduce_buf[block_sz / WARP_SIZE]  (for inter-sub-group reduction)
    const bool use_shared_A = (n <= MAX_N_SHARED);
    float * sA = shared_mem;
    float * sX = sA + (use_shared_A ? n * n : 0);
    float * reduce_buf = sX + n;

    // Load A into shared memory if it fits
    if (use_shared_A) {
        const int total_A = n * n;
        for (int i = tid; i < total_A; i += block_sz) {
            sA[i] = A_batch[i];
        }
    }

    // Load B column into sX
    for (int i = tid; i < n; i += block_sz) {
        sX[i] = B_batch[i * k + col_idx];
    }

    item.barrier(sycl::access::fence_space::local_space);

    // Forward substitution: for each row, compute
    //   x[row] = (b[row] - sum_{j<row} A[row,j]*x[j]) / A[row,row]
    //
    // Rows are processed sequentially (data dependency).
    // The dot product within each row is parallelized across threads.
    const int n_sub_groups = (block_sz + WARP_SIZE - 1) / WARP_SIZE;
    const int sub_group_id = tid / WARP_SIZE;
    const int lane_id      = tid % WARP_SIZE;

    for (int row = 0; row < n; ++row) {
        // Each thread computes a partial sum for columns [tid, tid+block_sz, ...]
        float partial = 0.0f;
        const float * A_row = use_shared_A ? (sA + row * n) : (A_batch + row * n);

        for (int j = tid; j < row; j += block_sz) {
            partial += A_row[j] * sX[j];
        }

        // Intra-sub-group reduction
        partial = warp_reduce_sum(partial);

        // Inter-sub-group reduction via shared memory
        if (lane_id == 0) {
            reduce_buf[sub_group_id] = partial;
        }
        item.barrier(sycl::access::fence_space::local_space);

        // First sub-group reduces the partial sums from all sub-groups
        if (sub_group_id == 0) {
            float val = (lane_id < n_sub_groups) ? reduce_buf[lane_id] : 0.0f;
            val = warp_reduce_sum(val);

            if (lane_id == 0) {
                float diag = use_shared_A ? sA[row * n + row] : A_batch[row * n + row];
                sX[row] = (sX[row] - val) / diag;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Write result back to global memory
    for (int i = tid; i < n; i += block_sz) {
        X_batch[i * k + col_idx] = sX[i];
    }
}

// --------------------------------------------------------------------------
// Host dispatch
// --------------------------------------------------------------------------
static void solve_tri_f32_sycl(
    const float * A,
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
    dpct::queue_ptr stream)
{
    const sycl::uint3 ne02_fd = init_fastdiv_values((uint32_t)ne02);
    const int total_batches = (int)(ne02 * ne03);
    if (total_batches == 0 || k == 0) {
        return;
    }

    // Choose block size based on n to balance parallelism vs overhead
    int block_size = WARP_SIZE;  // minimum useful size
    if (n > 128) {
        block_size = 256;
    } else if (n > 64) {
        block_size = 128;
    } else if (n > 32) {
        block_size = 64;
    } else if (n > 16) {
        block_size = 32;
    }
    // Ensure block_size is at least WARP_SIZE
    if (block_size < WARP_SIZE) {
        block_size = WARP_SIZE;
    }

    // Shared memory sizes (in floats)
    const bool use_shared_A = (n <= MAX_N_SHARED);
    const size_t shared_A_count = use_shared_A ? ((size_t)n * n) : 0;
    const size_t shared_X_count = (size_t)n;
    const size_t reduce_buf_count = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    const size_t total_shared_floats = shared_A_count + shared_X_count + reduce_buf_count;

    // Total work-groups: one per (batch, column) pair
    const int64_t n_work_groups = (int64_t)total_batches * k;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> shared_mem(
            sycl::range<1>(total_shared_floats), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(n_work_groups * block_size),
                sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item) {
                float * smem = shared_mem.get_multi_ptr<sycl::access::decorated::no>().get();
                solve_tri_f32_kernel(
                    A, B, X, ne02_fd,
                    nb02, nb03, nb12, nb13, nb2, nb3,
                    n, k, total_batches,
                    item, smem);
            });
    });
}

void ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0 = dst->src[0];  // A (n×n, lower triangular)
    const ggml_tensor * src1 = dst->src[1];  // B (n×k)

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

    solve_tri_f32_sycl(
        (const float *)src0->data,
        (const float *)src1->data,
        (float *)dst->data,
        (int)n, (int)k,
        ne02, ne03,
        src0->nb[2] / sizeof(float),
        src0->nb[3] / sizeof(float),
        src1->nb[2] / sizeof(float),
        src1->nb[3] / sizeof(float),
        dst->nb[2]  / sizeof(float),
        dst->nb[3]  / sizeof(float),
        stream);
}
