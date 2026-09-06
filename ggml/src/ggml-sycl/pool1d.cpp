#include "pool1d.hpp"

#include <cfloat>
#include <sycl/sycl.hpp>

namespace {
static inline int max_i32(int a, int b) { return a > b ? a : b; }
static inline int min_i32(int a, int b) { return a < b ? a : b; }

void pool1d_dispatch(sycl::queue & q, int iw, int ow, int kw, int sw, int pw, int parallel_elements, const float * src, float * dst, ggml_op_pool op) {
    const int block = 256;
    const int nblk = (parallel_elements + block - 1) / block;

    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
            [=](sycl::nd_item<1> item) {
                const int idx = (int) item.get_global_id(0);
                if (idx >= parallel_elements) {
                    return;
                }

                const int nc = idx / ow;
                const int cur_ow = idx % ow;

                const float * i_ptr = src + nc * iw;
                float * o_ptr = dst + nc * ow;

                const int start = cur_ow * sw - pw;
                const int b = max_i32(0, start);
                const int e = min_i32(iw, start + kw);

                float res;
                switch (op) {
                    case GGML_OP_POOL_AVG: res = 0.0f; break;
                    case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
                    default: return;
                }

                int count = 0;
                for (int i = b; i < e; ++i) {
                    const float cur = i_ptr[i];
                    switch (op) {
                        case GGML_OP_POOL_AVG: res += cur; break;
                        case GGML_OP_POOL_MAX: res = sycl::fmax(res, cur); break;
                        default: break;
                    }
                    ++count;
                }

                if (op == GGML_OP_POOL_AVG) {
                    res = count > 0 ? (res / count) : 0.0f;
                }

                o_ptr[cur_ow] = res;
            });
}
}

void ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const ggml_op_pool op = (ggml_op_pool) ggml_get_op_params_i32(dst, 0);
    const int k0 = ggml_get_op_params_i32(dst, 1);
    const int s0 = ggml_get_op_params_i32(dst, 2);
    const int p0 = ggml_get_op_params_i32(dst, 3);

    const int iw = (int) src0->ne[0];
    const int ow = (int) dst->ne[0];
    const int nr = (int) ggml_nrows(src0);
    const int parallel_elements = nr * ow;

    pool1d_dispatch(ctx.stream(), iw, ow, k0, s0, p0, parallel_elements, (const float *) src0->data, (float *) dst->data, op);
}

bool ggml_sycl_supports_pool1d(const ggml_tensor * op) {
    return ggml_is_contiguous(op->src[0]) &&
           op->src[0]->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32;
}
