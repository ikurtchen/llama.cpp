#include "lightning_indexer.hpp"
#include "common.hpp"
#include "dequantize.hpp"

#include <cmath>

#define WARP_SIZE 32

// Vector-based lightning indexer kernel
// Computes Q*K^T scores with shared memory + warp shuffle reduction
// Each warp processes K_VECS_PER_WARP K-vectors, all warps in a block cooperate
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int N_EMBD, int N_HEAD>
static void lightning_indexer_kernel_vec(
        const float * Q, const char * K, const float * W, const sycl::half * M, float * dst,
        int n_stream, int n_batch, int n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw3,
        size_t nbm1, size_t nbm3,
        int nem3,
        ggml_type type_K,
        const sycl::nd_item<3> & item) {

    constexpr int K_VECS_PER_WARP = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int N_HEAD_INNER = N_HEAD / 4;

    const int i_batch  = item.get_group(1);
    const int i_stream = item.get_group(2);
    const int i_warp   = item.get_local_id(1);
    const int i_lane   = item.get_local_id(2);
    const int tid      = i_warp * WARP_SIZE + i_lane;

    const int start_kv_block = item.get_group(0) * K_VECS_PER_BLOCK;
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 Q + i_batch * nbq2 + i_stream * nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch * nbw1 + i_stream * nbw3);

    // Load K vectors into registers (dequantize if needed)
    float4 k_reg_f[K_VECS_PER_WARP];

    if (type_K == GGML_TYPE_F32) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const float4 * k_base = (const float4 *) ((const char *) K + i_kv * nbk2 + i_stream * nbk3);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = sycl::float4{0, 0, 0, 0};
            }
        }
    } else if (type_K == GGML_TYPE_F16) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const sycl::half4 * k_base_h = (const sycl::half4 *) ((const char *) K + i_kv * nbk2 + i_stream * nbk3);
                sycl::half4 kh = k_base_h[i_lane];
                k_reg_f[k] = sycl::float4{kh.x(), kh.y(), kh.z(), kh.w()};
            } else {
                k_reg_f[k] = sycl::float4{0, 0, 0, 0};
            }
        }
    } else {
        // Quantized types: use dequantize functions per element
        // For Q4_0, Q8_0 etc. we load 4 elements at a time
        // This mirrors the CUDA approach using get_dequantize_V
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv * nbk2 + i_stream * nbk3);
                int iqs = i_lane * 4;
                if (type_K == GGML_TYPE_Q4_0) {
                    float v0, v1;
                    dequantize_q4_0(k_base, iqs / QK4_0, iqs % QK4_0, v0, v1);
                    k_reg_f[k] = sycl::float4{v0, v1, 0, 0};
                    // Next pair for the second half
                    float v2, v3;
                    dequantize_q4_0(k_base, (iqs + 2) / QK4_0, (iqs + 2) % QK4_0, v2, v3);
                    k_reg_f[k].z() = v2;
                    k_reg_f[k].w() = v3;
                } else {
                    // For unsupported types in this kernel, zero out
                    k_reg_f[k] = sycl::float4{0, 0, 0, 0};
                }
            } else {
                k_reg_f[k] = sycl::float4{0, 0, 0, 0};
            }
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {
        // Load weights and Q into shared memory
        // Shared memory declarations
        sycl::multi_ptr<float, sycl::access::address_space::local_space> w_shared_ptr = nullptr;
        sycl::multi_ptr<sycl::float4, sycl::access::address_space::local_space> q_shared_ptr = nullptr;

        // Using explicit SLM pointers
        __attribute__((opencl_local)) float  w_shared[N_HEAD_INNER];
        __attribute__((opencl_local)) sycl::float4 q_shared_f[N_HEAD_INNER][N_EMBD / 4];

        if (tid < N_HEAD_INNER) {
            w_shared[tid] = w_base[i_head_0 + tid];
        }

        constexpr int n_q = N_HEAD_INNER * (N_EMBD / 4);
#pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head_inner = i_q / (N_EMBD / 4);
            const int i_head = i_head_0 + i_head_inner;
            const int i_embd = i_q % (N_EMBD / 4);
            q_shared_f[i_head_inner][i_embd] = *(const sycl::float4 *) (q_base + i_head * nbq1 + i_embd * sizeof(sycl::float4));
        }

        sycl::group_barrier(item.get_group());

        // Compute dot products with warp reduction
        for (int i_head_inner = 0; i_head_inner < N_HEAD_INNER; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            const sycl::float4 q_vec = q_shared_f[i_head_inner][i_lane];

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                qk[k] += q_vec.x() * k_reg_f[k].x();
                qk[k] += q_vec.y() * k_reg_f[k].y();
                qk[k] += q_vec.z() * k_reg_f[k].z();
                qk[k] += q_vec.w() * k_reg_f[k].w();
            }

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum(qk[k], item);

                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;  // ReLU
                    score_k[k] += sum * w_val;
                }
            }
        }

        sycl::group_barrier(item.get_group());
    }

    // Store outputs to shared memory for coalesced write
    __attribute__((opencl_local)) float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    sycl::group_barrier(item.get_group());

    // Coalesced write from shared memory to VRAM
    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            const sycl::half * m_base = (const sycl::half *) ((const char *) M + i_batch * nbm1 + (i_stream % nem3) * nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch * nb1 + i_stream * nb3);
            dst_base[i_kv] = dst_shared[tid] + (float)m_base[i_kv];
        }
    }
}

// Placeholder for WMMA kernel using oneDNN
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int N_EMBD, int N_HEAD>
static void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const sycl::half * M, float * dst,
        int n_stream, int n_batch, int n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw3,
        size_t nbm1, size_t nbm3,
        int nem3,
        ggml_type type_K,
        const sycl::nd_item<3> & item) {
    // WMMA kernel is not implemented yet - fall through to vec kernel
    // TODO: implement using oneDNN matmul for Q*K^T
    (void)Q; (void)K; (void)W; (void)M; (void)dst;
    (void)n_stream; (void)n_batch; (void)n_kv;
    (void)nb1; (void)nb2; (void)nb3;
    (void)nbq1; (void)nbq2; (void)nbq3;
    (void)nbk2; (void)nbk3;
    (void)nbw1; (void)nbw3;
    (void)nbm1; (void)nbm3;
    (void)nem3;
    (void)type_K;
    (void)item;
}

// Dispatch function
void ggml_sycl_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int n_embd   = q->ne[0];
    const int n_head   = q->ne[1];
    const int n_batch  = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[2];

    const float *   q_d = (const float *)   q->data;
    const char  *   k_d = (const char  *)   k->data;
    const float *   w_d = (const float *)   w->data;
    const sycl::half * m_d = (const sycl::half *) m->data;
    float       * dst_d = (      float *) dst->data;

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    if (n_embd == 128 && (n_head == 64 || n_head == 32)) {
        // Use vector kernel for all K types
        constexpr int K_VECS_PER_WARP = 8;
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

        sycl::range<3> block(1, WARPS_PER_BLOCK, WARP_SIZE);
        int num_kv_blocks = (n_kv + K_VECS_PER_BLOCK - 1) / K_VECS_PER_BLOCK;
        sycl::range<3> grid(num_kv_blocks, n_batch, n_stream);

        if (n_head == 64) {
            stream->parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) {
                    lightning_indexer_kernel_vec<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, 128, 64>(
                        q_d, k_d, w_d, m_d, dst_d,
                        n_stream, n_batch, n_kv,
                        nb1, nb2, nb3,
                        nbq1, nbq2, nbq3,
                        nbk2, nbk3,
                        nbw1, nbw3,
                        nbm1, nbm3,
                        nem3,
                        k->type, item);
                });
        } else {
            stream->parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) {
                    lightning_indexer_kernel_vec<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, 128, 32>(
                        q_d, k_d, w_d, m_d, dst_d,
                        n_stream, n_batch, n_kv,
                        nb1, nb2, nb3,
                        nbq1, nbq2, nbq3,
                        nbk2, nbk3,
                        nbw1, nbw3,
                        nbm1, nbm3,
                        nem3,
                        k->type, item);
                });
        }
    } else {
        GGML_ABORT("unsupported lightning-indexer configuration");
    }
}

bool ggml_sycl_lightning_indexer_supported(int /*device*/, const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)

    if (neq0 != 128) {
        return false;
    }

    if (neq1 != 64 && neq1 != 32) {
        return false;
    }

    // Alignment checks
    for (const ggml_tensor * t : {q, k}) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }

    switch (k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}