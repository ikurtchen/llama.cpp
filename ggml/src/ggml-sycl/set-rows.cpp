// SYCL set_rows op: dst[idx[i01,i11,i12], i02, i03, :] = cast(src0[i01, i02, i03, :]) for each
// row i01, with src1 (int32/int64 row indices) broadcast across dst's higher batch dims when
// src1 has fewer batch entries than dst (ne11/ne12 divide ne02/ne03). Mirrors
// ggml-cuda/set-rows.cu's non-quantized (F32/F16 dst) path but written directly against SYCL.
#include "set-rows.hpp"

#include <sycl/sycl.hpp>

namespace {
template <typename T>
T cast_from_f32(float v);
template <> float cast_from_f32<float>(float v) { return v; }
template <> sycl::half cast_from_f32<sycl::half>(float v) { return (sycl::half) v; }

template <typename src_t, typename idx_t, typename dst_t>
void set_rows_impl(sycl::queue & q, const src_t * src0, const idx_t * src1, dst_t * dst,
                    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                    int64_t ne11, int64_t ne12,
                    int64_t s01, int64_t s02, int64_t s03,
                    int64_t s10, int64_t s11, int64_t s12,
                    int64_t s1, int64_t s2, int64_t s3) {
    q.parallel_for(sycl::range<3>((size_t) (ne02 * ne03), (size_t) ne01, (size_t) ne00), [=](sycl::item<3> item) {
        const int64_t i02_i03 = item.get_id(0);
        const int64_t i01     = item.get_id(1);
        const int64_t i00     = item.get_id(2);
        const int64_t i03     = i02_i03 / ne02;
        const int64_t i02     = i02_i03 % ne02;

        const int64_t i12 = i03 % ne12;
        const int64_t i11 = i02 % ne11;
        const int64_t i10 = i01;

        const int64_t dst_row = (int64_t) src1[i10 * s10 + i11 * s11 + i12 * s12];

        const src_t * src0_row = src0 + i01 * s01 + i02 * s02 + i03 * s03;
        dst_t       * dst_row_ptr = dst + dst_row * s1 + i02 * s2 + i03 * s3;

        dst_row_ptr[i00] = cast_from_f32<dst_t>((float) src0_row[i00]);
    });
}
}  // namespace

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    GGML_TENSOR_BINARY_OP_LOCALS

    sycl::queue & q = ctx.stream();

#define SET_ROWS_DISPATCH(SRC_T, IDX_T, DST_T)                                                                  \
    set_rows_impl<SRC_T, IDX_T, DST_T>(q, (const SRC_T *) src0->data, (const IDX_T *) src1->data,               \
        (DST_T *) dst->data, ne00, ne01, ne02, ne03, ne11, ne12,                                                \
        nb01 / sizeof(SRC_T), nb02 / sizeof(SRC_T), nb03 / sizeof(SRC_T),                                       \
        nb10 / sizeof(IDX_T), nb11 / sizeof(IDX_T), nb12 / sizeof(IDX_T),                                       \
        nb1 / sizeof(DST_T), nb2 / sizeof(DST_T), nb3 / sizeof(DST_T))

    const bool idx64 = src1->type == GGML_TYPE_I64;
    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (idx64) { SET_ROWS_DISPATCH(float, int64_t, float); } else { SET_ROWS_DISPATCH(float, int32_t, float); }
    } else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        if (idx64) { SET_ROWS_DISPATCH(float, int64_t, sycl::half); } else { SET_ROWS_DISPATCH(float, int32_t, sycl::half); }
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        if (idx64) { SET_ROWS_DISPATCH(sycl::half, int64_t, sycl::half); } else { SET_ROWS_DISPATCH(sycl::half, int32_t, sycl::half); }
    } else {
        GGML_ABORT("ggml_sycl_op_set_rows: unsupported src0/dst type combination");
    }
#undef SET_ROWS_DISPATCH
}

bool ggml_sycl_supports_set_rows(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const bool idx_ok = src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32;
    const bool dst_ok  = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
    const bool src0_ok = src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16);
    return idx_ok && dst_ok && src0_ok;
}
