#include "pool2d.hpp"

#include <cfloat>
#include <sycl/sycl.hpp>

namespace {
static inline int max_i32(int a, int b) { return a > b ? a : b; }
static inline int min_i32(int a, int b) { return a < b ? a : b; }

void pool2d_dispatch(
        sycl::queue & q,
        int ih,
        int iw,
        int oh,
        int ow,
        int kh,
        int kw,
        int sh,
        int sw,
        int ph,
        int pw,
        int parallel_elements,
        const float * src,
        float * dst,
        ggml_op_pool op) {
    const int block = 256;
    const int nblk = (parallel_elements + block - 1) / block;

    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
            [=](sycl::nd_item<1> item) {
                const int idx = (int) item.get_global_id(0);
                if (idx >= parallel_elements) {
                    return;
                }

                const int i_hw = ih * iw;
                const int o_hw = oh * ow;
                const int nc = idx / o_hw;
                const int cur_oh = (idx % o_hw) / ow;
                const int cur_ow = idx % ow;
                const float * i_ptr = src + nc * i_hw;
                float * o_ptr = dst + nc * o_hw;
                const int start_h = cur_oh * sh - ph;
                const int bh = max_i32(0, start_h);
                const int eh = min_i32(ih, start_h + kh);
                const int start_w = cur_ow * sw - pw;
                const int bw = max_i32(0, start_w);
                const int ew = min_i32(iw, start_w + kw);
                const float scale = 1.0f / (kh * kw);

                float res;
                switch (op) {
                    case GGML_OP_POOL_AVG: res = 0.0f; break;
                    case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
                    default: return;
                }

                for (int i = bh; i < eh; ++i) {
                    for (int j = bw; j < ew; ++j) {
                        const float cur = i_ptr[i * iw + j];
                        switch (op) {
                            case GGML_OP_POOL_AVG: res += cur * scale; break;
                            case GGML_OP_POOL_MAX: res = sycl::fmax(res, cur); break;
                            default: break;
                        }
                    }
                }

                o_ptr[cur_oh * ow + cur_ow] = res;
            });
}
}

void ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const ggml_op_pool op = (ggml_op_pool) ggml_get_op_params_i32(dst, 0);
    const int k0 = ggml_get_op_params_i32(dst, 1);
    const int k1 = ggml_get_op_params_i32(dst, 2);
    const int s0 = ggml_get_op_params_i32(dst, 3);
    const int s1 = ggml_get_op_params_i32(dst, 4);
    const int p0 = ggml_get_op_params_i32(dst, 5);
    const int p1 = ggml_get_op_params_i32(dst, 6);

    const int ih = (int) src0->ne[1];
    const int iw = (int) src0->ne[0];
    const int n = (int) dst->ne[3];
    const int oc = (int) dst->ne[2];
    const int oh = (int) dst->ne[1];
    const int ow = (int) dst->ne[0];
    const int parallel_elements = n * oc * oh * ow;

    pool2d_dispatch(ctx.stream(), ih, iw, oh, ow, k1, k0, s1, s0, p1, p0, parallel_elements,
            (const float *) src0->data, (float *) dst->data, op);
}

bool ggml_sycl_supports_pool2d(const ggml_tensor * op) {
    return ggml_is_contiguous(op->src[0]) &&
           op->src[0]->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32;
}
