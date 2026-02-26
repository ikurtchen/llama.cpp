#include <sycl/sycl.hpp>
#include "wkv.hpp"

constexpr int WKV_BLOCK_SIZE = 64;  // Matching CUDA_WKV_BLOCK_SIZE

// Helper function for the main kernel
template <int block_size>
static void rwkv_wkv6_f32_kernel(
    const int B, const int T, const int C, const int H,
    const float* k, const float* v, const float* r,
    const float* tf, const float* td, const float* s,
    float* dst, const sycl::nd_item<3>& item_ct1, float* shared_mem) {

    const int tid = item_ct1.get_local_id(2);
    const int bid = item_ct1.get_group(2);

    const int head_size = block_size;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    // Set up shared memory pointers
    float* _k = shared_mem;
    float* _r = _k + head_size;
    float* _tf = _r + head_size;
    float* _td = _tf + head_size;

    // Local state array
    float state[block_size];

    // Load initial state
    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    // Sync threads before shared memory operations
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Load time-mixing parameters
    _tf[tid] = tf[head_i * head_size + tid];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main sequence processing loop
    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
         t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Load current timestep data to shared memory
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        const float _v = v[t];
        float y = 0;

        #pragma unroll
        for (int j = 0; j < head_size; j += 4) {
            // Optimization: Use true vector loads from SLM instead of scalar construction.
            // This performs a single 128-bit aligned load from shared memory instead of
            // 4 individual 32-bit reads. Reference: hw_spec_b60.md (SIMD16, 456 GB/s bandwidth)
            // and optimization_guide.md Section 2.2 (vectorized memory access).
            // Note: state[] is in register, not SLM, so keep float4 construction for state.
            const sycl::float4 k4 = *reinterpret_cast<const sycl::float4*>(&_k[j]);
            const sycl::float4 r4 = *reinterpret_cast<const sycl::float4*>(&_r[j]);
            const sycl::float4 tf4 = *reinterpret_cast<const sycl::float4*>(&_tf[j]);
            const sycl::float4 td4 = *reinterpret_cast<const sycl::float4*>(&_td[j]);
            sycl::float4 s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);

            // Compute key-value product
            sycl::float4 kv4 = k4 * _v;

            // Accumulate weighted sum
            y += sycl::dot(r4, tf4 * kv4 + s4);

            // Update state
            s4 = s4 * td4 + kv4;

            // Store updated state
            state[j] = s4.x();
            state[j+1] = s4.y();
            state[j+2] = s4.z();
            state[j+3] = s4.w();
        }

        dst[t] = y;
    }

    // Save final state
    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

template <int block_size>
static void rwkv_wkv7_f32_kernel(
    const int B, const int T, const int C, const int H,
    const float* r, const float* w, const float* k, const float* v,
    const float* a, const float* b, const float* s,
    float* dst, const sycl::nd_item<3>& item_ct1, float* shared_mem) {

    const int tid = item_ct1.get_local_id(2);
    const int bid = item_ct1.get_group(2);

    const int head_size = block_size;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float* _r = shared_mem;
    float* _w = _r + head_size;
    float* _k = _w + head_size;
    float* _a = _k + head_size;
    float* _b = _a + head_size;

    float state[block_size];

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
    }

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
         t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        const float _v = v[t];
        float y = 0, sa = 0;

        #pragma unroll
        for (int j = 0; j < head_size; j += 4) {
            const sycl::float4 a4 = *reinterpret_cast<const sycl::float4*>(&_a[j]);
            sycl::float4 s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa += sycl::dot(a4, s4);
        }

        #pragma unroll
        for (int j = 0; j < head_size; j += 4) {
            const sycl::float4 r4 = *reinterpret_cast<const sycl::float4*>(&_r[j]);
            const sycl::float4 w4 = *reinterpret_cast<const sycl::float4*>(&_w[j]);
            const sycl::float4 k4 = *reinterpret_cast<const sycl::float4*>(&_k[j]);
            const sycl::float4 b4 = *reinterpret_cast<const sycl::float4*>(&_b[j]);
            sycl::float4 s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);

            sycl::float4 kv4 = k4 * _v;

            s4 = s4 * w4 + kv4 + sa * b4;
            y += sycl::dot(r4, s4);

            state[j] = s4.x();
            state[j+1] = s4.y();
            state[j+2] = s4.z();
            state[j+3] = s4.w();
        }

        dst[t] = y;
    }

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
    }
}

void ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    const float* k_d = (const float*)dst->src[0]->data;
    const float* v_d = (const float*)dst->src[1]->data;
    const float* r_d = (const float*)dst->src[2]->data;
    const float* tf_d = (const float*)dst->src[3]->data;
    const float* td_d = (const float*)dst->src[4]->data;
    const float* s_d = (const float*)dst->src[5]->data;
    float* dst_d = (float*)dst->data;

    const int64_t B = dst->src[5]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_SIZE || C / H == WKV_BLOCK_SIZE * 2); // The current sycl kernel is designed for RWKV6, HEAD_SIZE == 64

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    const size_t shared_mem_size = C / H * 4; // For k, r, tf, td
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    if (C / H == WKV_BLOCK_SIZE) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_SIZE>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (float*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } else {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_SIZE * 2>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (float*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}

void ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    const float* r_d = (const float*)dst->src[0]->data;
    const float* w_d = (const float*)dst->src[1]->data;
    const float* k_d = (const float*)dst->src[2]->data;
    const float* v_d = (const float*)dst->src[3]->data;
    const float* a_d = (const float*)dst->src[4]->data;
    const float* b_d = (const float*)dst->src[5]->data;
    const float* s_d = (const float*)dst->src[6]->data;
    float* dst_d = (float*)dst->data;

    const int64_t B = dst->src[6]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_SIZE || C / H == WKV_BLOCK_SIZE * 2);

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    const size_t shared_mem_size = C / H * 5; // For r, w, k, a, b
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    if (C / H == WKV_BLOCK_SIZE) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_SIZE>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (float*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } else {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_SIZE * 2>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (float*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}
