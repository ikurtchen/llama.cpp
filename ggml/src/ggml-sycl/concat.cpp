// SYCL concat op: concatenate src0 and src1 along dimension `dim` (op_params[0]) into dst.
// Uses a general strided element-copy kernel (byte-addressed by element size) so it handles
// contiguous and non-contiguous tensors uniformly. Mirrors ggml-cuda/concat.cu's algorithm
// but written directly against SYCL; the CUDA fast-path contiguous kernels are not ported
// since correctness (not peak throughput) is the goal of this migration step.
#include "concat.hpp"

#include <sycl/sycl.hpp>

static void concat_strided(sycl::queue & q, const char * src0, const char * src1, char * dst, int dim,
                            int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                            uint64_t nb00, uint64_t nb01, uint64_t nb02, uint64_t nb03,
                            uint64_t nb10, uint64_t nb11, uint64_t nb12, uint64_t nb13,
                            int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                            uint64_t nb0, uint64_t nb1, uint64_t nb2, uint64_t nb3,
                            size_t type_size) {
    q.parallel_for(sycl::range<3>(ne3, ne2, ne1), [=](sycl::item<3> item) {
        const int64_t i3 = item.get_id(0);
        const int64_t i2 = item.get_id(1);
        const int64_t i1 = item.get_id(2);

        for (int64_t i0 = 0; i0 < ne0; i0++) {
            const char * x;
            if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                x = src0 + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00;
            } else {
                switch (dim) {
                    case 0:  x = src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10; break;
                    case 1:  x = src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10; break;
                    case 2:  x = src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10; break;
                    default: x = src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10; break;
                }
            }
            char * y = dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0;
            for (size_t b = 0; b < type_size; b++) {
                y[b] = x[b];
            }
        }
    });
}

void ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(dst->type == src0->type);
    GGML_ASSERT(ggml_blck_size(src0->type) == 1);

    const int32_t dim = ((const int32_t *) dst->op_params)[0];

    sycl::queue & q = ctx.stream();
    concat_strided(q, (const char *) src0->data, (const char *) src1->data, (char *) dst->data, dim,
                   src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                   src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                   src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                   dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                   dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                   ggml_type_size(src0->type));
}

bool ggml_sycl_supports_concat(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return src0->type == src1->type && op->type == src0->type && ggml_blck_size(src0->type) == 1;
}
