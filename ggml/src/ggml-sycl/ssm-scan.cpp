#include "ssm-scan.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace {

static inline float softplus_f32(float x) {
    return x <= 20.0f ? sycl::log(1.0f + sycl::exp(x)) : x;
}

template <int D_STATE>
static void ssm_scan_f32(
        sycl::queue & q,
        const float * src0,
        const float * src1,
        const float * src2,
        const float * src3,
        const float * src4,
        const float * src5,
        const int32_t * src6,
        float * dst,
        size_t src0_nb1,
        size_t src0_nb2,
        size_t src0_nb3,
        size_t src1_nb1,
        size_t src1_nb2,
        size_t src1_nb3,
        size_t src2_nb1,
        size_t src2_nb2,
        size_t src3_nb1,
        size_t src4_nb1,
        size_t src4_nb2,
        size_t src4_nb3,
        size_t src5_nb1,
        size_t src5_nb2,
        size_t src5_nb3,
        int64_t n_head,
        int64_t head_dim,
        int64_t n_group,
        int64_t n_tok,
        int64_t n_seq,
        int64_t K) {
    const int64_t d_inner = n_head * head_dim;
    const int64_t y_elems = d_inner * n_tok * n_seq;

    q.parallel_for(
            sycl::nd_range<3>(sycl::range<3>((size_t) n_seq, (size_t) n_head, (size_t) head_dim * (size_t) D_STATE), sycl::range<3>(1, 1, (size_t) D_STATE)),
            [=](sycl::nd_item<3> item) {
                const int seq = (int) item.get_group(0);
                const int head = (int) item.get_group(1);
                const int dim = (int) item.get_group(2);
                const int row = (int) item.get_local_id(2);
                const auto group = item.get_group();

                const int group_idx = head / (int) (n_head / n_group);
                const int32_t state_seq = src6[seq];

                const char * s_base = reinterpret_cast<const char *>(src0) + (int64_t) state_seq * src0_nb3 + (int64_t) head * src0_nb2 + (int64_t) dim * src0_nb1;
                float state_val = reinterpret_cast<const float *>(s_base)[row];

                float * state_out_base = dst + y_elems;
                const float * A_row = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src3) + (int64_t) head * src3_nb1);
                const bool A_broadcast = src3_nb1 == sizeof(float);

                for (int token = 0; token < n_tok; ++token) {
                    const float x_val = *reinterpret_cast<const float *>(reinterpret_cast<const char *>(src1) + (int64_t) seq * src1_nb3 + (int64_t) token * src1_nb2 + (int64_t) head * src1_nb1 + (int64_t) dim * sizeof(float));
                    const float dt_val = *reinterpret_cast<const float *>(reinterpret_cast<const char *>(src2) + (int64_t) seq * src2_nb2 + (int64_t) token * src2_nb1 + (int64_t) head * sizeof(float));
                    const float B_val = *reinterpret_cast<const float *>(reinterpret_cast<const char *>(src4) + (int64_t) seq * src4_nb3 + (int64_t) token * src4_nb2 + (int64_t) group_idx * src4_nb1 + (int64_t) row * sizeof(float));
                    const float C_val = *reinterpret_cast<const float *>(reinterpret_cast<const char *>(src5) + (int64_t) seq * src5_nb3 + (int64_t) token * src5_nb2 + (int64_t) group_idx * src5_nb1 + (int64_t) row * sizeof(float));

                    const float dt_sp = softplus_f32(dt_val);
                    const float A_val = A_broadcast ? A_row[0] : A_row[row];
                    const float dA = sycl::exp(dt_sp * A_val);
                    const float x_dt = x_val * dt_sp;

                    state_val = state_val * dA + B_val * x_dt;
                    const float partial = state_val * C_val;
                    const float sum = sycl::reduce_over_group(group, partial, sycl::plus<float>());

                    if (row == 0) {
                        dst[((int64_t) seq * n_tok + token) * d_inner + (int64_t) head * head_dim + dim] = sum;
                    }

                    if (K > 1) {
                        const int64_t slot = n_tok - 1 - token;
                        if (slot >= 0 && slot < K) {
                            char * slot_base = reinterpret_cast<char *>(state_out_base) + slot * n_seq * src0_nb3 + (int64_t) seq * src0_nb3 + (int64_t) head * src0_nb2 + (int64_t) dim * src0_nb1;
                            reinterpret_cast<float *>(slot_base)[row] = state_val;
                        }
                    }
                }

                if (K == 1) {
                    char * slot0 = reinterpret_cast<char *>(state_out_base) + (int64_t) seq * src0_nb3 + (int64_t) head * src0_nb2 + (int64_t) dim * src0_nb1;
                    reinterpret_cast<float *>(slot0)[row] = state_val;
                }
            });
}

} // namespace

void ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    const ggml_tensor * src3 = dst->src[3];
    const ggml_tensor * src4 = dst->src[4];
    const ggml_tensor * src5 = dst->src[5];
    const ggml_tensor * src6 = dst->src[6];

    const int64_t d_state = src0->ne[0];
    const int64_t head_dim = src0->ne[1];
    const int64_t n_head = src1->ne[1];
    const int64_t n_group = src4->ne[1];
    const int64_t n_tok = src1->ne[2];
    const int64_t n_seq = src1->ne[3];
    const int32_t K_param = ggml_get_op_params_i32(dst, 0);
    const int64_t K = K_param > 0 ? K_param : 1;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);
    GGML_ASSERT(src3->type == GGML_TYPE_F32);
    GGML_ASSERT(src4->type == GGML_TYPE_F32);
    GGML_ASSERT(src5->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    sycl::queue & q = ctx.stream();
    switch (d_state) {
        case 16:
            ssm_scan_f32<16>(q, (const float *) src0->data, (const float *) src1->data, (const float *) src2->data,
                             (const float *) src3->data, (const float *) src4->data, (const float *) src5->data,
                             (const int32_t *) src6->data, (float *) dst->data,
                             src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[1], src1->nb[2], src1->nb[3],
                             src2->nb[1], src2->nb[2], src3->nb[1], src4->nb[1], src4->nb[2], src4->nb[3],
                             src5->nb[1], src5->nb[2], src5->nb[3], n_head, head_dim, n_group, n_tok, n_seq, K);
            break;
        case 128:
            ssm_scan_f32<128>(q, (const float *) src0->data, (const float *) src1->data, (const float *) src2->data,
                              (const float *) src3->data, (const float *) src4->data, (const float *) src5->data,
                              (const int32_t *) src6->data, (float *) dst->data,
                              src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[1], src1->nb[2], src1->nb[3],
                              src2->nb[1], src2->nb[2], src3->nb[1], src4->nb[1], src4->nb[2], src4->nb[3],
                              src5->nb[1], src5->nb[2], src5->nb[3], n_head, head_dim, n_group, n_tok, n_seq, K);
            break;
        case 256:
            ssm_scan_f32<256>(q, (const float *) src0->data, (const float *) src1->data, (const float *) src2->data,
                              (const float *) src3->data, (const float *) src4->data, (const float *) src5->data,
                              (const int32_t *) src6->data, (float *) dst->data,
                              src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[1], src1->nb[2], src1->nb[3],
                              src2->nb[1], src2->nb[2], src3->nb[1], src4->nb[1], src4->nb[2], src4->nb[3],
                              src5->nb[1], src5->nb[2], src5->nb[3], n_head, head_dim, n_group, n_tok, n_seq, K);
            break;
        default:
            GGML_ABORT("ssm scan requires d_state 16, 128, or 256");
    }
}

bool ggml_sycl_supports_ssm_scan(const ggml_tensor * op) {
    if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32 || op->src[2]->type != GGML_TYPE_F32 ||
        op->src[3]->type != GGML_TYPE_F32 || op->src[4]->type != GGML_TYPE_F32 || op->src[5]->type != GGML_TYPE_F32 ||
        op->src[6]->type != GGML_TYPE_I32 || op->type != GGML_TYPE_F32) {
        return false;
    }

    const int32_t K = ggml_get_op_params_i32(op, 0);
    if (op->src[3]->ne[0] == 1) {
        return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % 16 == 0;
    }

    if (K > 1) {
        return false;
    }

    return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
}
