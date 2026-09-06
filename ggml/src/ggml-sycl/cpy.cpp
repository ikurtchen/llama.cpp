// SYCL cpy/dup op: element-wise copy between two (possibly strided, possibly different-dtype)
// tensors. Mirrors ggml-cuda/cpy.cu's index-unflattening scheme (flatten dst index -> unflatten
// against src0's ne/nb, then against src1/dst's ne/nb) but is written directly against SYCL
// nd_range, not translated from the CUDA source.
#include "cpy.hpp"
#include "common.hpp"

#include <sycl/sycl.hpp>

template <typename src_t, typename dst_t>
static void ggml_sycl_op_cpy_impl(sycl::queue & q, const char * cx, char * cdst, int64_t ne,
                                   int64_t ne00, int64_t ne01, int64_t ne02,
                                   int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
                                   int64_t ne10, int64_t ne11, int64_t ne12,
                                   int64_t nb10, int64_t nb11, int64_t nb12, int64_t nb13) {
    const int64_t block = 256;
    const int64_t nblk  = (ne + block - 1) / block;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int64_t i = item.get_global_id(0);
        if (i >= ne) {
            return;
        }

        const int64_t i03 = i / (ne00 * ne01 * ne02);
        const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
        const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
        const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
        const int64_t src_off = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

        const int64_t i13 = i / (ne10 * ne11 * ne12);
        const int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
        const int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
        const int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
        const int64_t dst_off = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

        const src_t val = *reinterpret_cast<const src_t *>(cx + src_off);
        *reinterpret_cast<dst_t *>(cdst + dst_off) = (dst_t) val;
    });
}

template <typename dst_t>
static void dispatch_src(sycl::queue & q, const ggml_tensor * src0, const char * cx, char * cdst, int64_t ne,
                          int64_t ne00, int64_t ne01, int64_t ne02, int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
                          int64_t ne10, int64_t ne11, int64_t ne12, int64_t nb10, int64_t nb11, int64_t nb12, int64_t nb13) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_sycl_op_cpy_impl<float, dst_t>(q, cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
            break;
        case GGML_TYPE_F16:
            ggml_sycl_op_cpy_impl<sycl::half, dst_t>(q, cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
            break;
        default:
            GGML_ABORT("%s: unsupported src type for SYCL cpy: %s", __func__, ggml_type_name(src0->type));
    }
}

void ggml_sycl_cpy(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, ggml_tensor * dst) {
    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(dst));

    sycl::queue & q = ctx.stream();

    const int64_t ne   = ggml_nelements(src0);
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb);
    GGML_TENSOR_LOCALS(int64_t, ne1, dst,  ne);
    GGML_TENSOR_LOCALS(size_t,  nb1, dst,  nb);

    const char * cx   = (const char *) src0->data;
    char *       cdst = (char *) dst->data;

    switch (dst->type) {
        case GGML_TYPE_F32:
            dispatch_src<float>(q, src0, cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
            break;
        case GGML_TYPE_F16:
            dispatch_src<sycl::half>(q, src0, cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type for SYCL cpy: %s", __func__, ggml_type_name(dst->type));
    }
}

bool ggml_sycl_supports_cpy(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    auto is_supported_type = [](ggml_type t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16; };
    return is_supported_type(src0->type) && is_supported_type(op->type);
}
