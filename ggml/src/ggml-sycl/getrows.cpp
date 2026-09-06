// SYCL get_rows mirrors ggml-cuda/getrows.cu's gather + in-kernel dequant path.
// This is a hand-written SYCL 2020 port for dense F32/F16 and Q8_0/Q4_0/Q4_1 rows.
#include "getrows.hpp"

#define GGML_COMMON_DECL_SYCL
#include "ggml-common.h"

#include <sycl/sycl.hpp>

namespace {

template <typename dst_t>
static inline dst_t cast_from_f32(float v);

template <>
inline float cast_from_f32<float>(float v) {
    return v;
}

template <>
inline sycl::half cast_from_f32<sycl::half>(float v) {
    return (sycl::half) v;
}

template <typename src_t, typename dst_t>
static void get_rows_dense(sycl::queue & q, const src_t * src0, const int32_t * rows, dst_t * dst,
                           int64_t ne00, int64_t ne10, int64_t ne11, int64_t ne12,
                           size_t nb01, size_t nb02, size_t nb03,
                           size_t nb1, size_t nb2, size_t nb3) {
    const size_t s1    = nb1 / sizeof(dst_t);
    const size_t s2    = nb2 / sizeof(dst_t);
    const size_t s3    = nb3 / sizeof(dst_t);

    q.parallel_for(sycl::range<3>((size_t) ne12, (size_t) ne11, (size_t) ne10), [=](sycl::item<3> item) {
        const int64_t i12 = item.get_id(0);
        const int64_t i11 = item.get_id(1);
        const int64_t i10 = item.get_id(2);
        const int64_t row_idx = i10 + ne10 * (i11 + ne11 * i12);
        const int32_t src_row = rows[row_idx];
        const char * src0_row = (const char *) src0 + (ptrdiff_t) src_row * (ptrdiff_t) nb01 + (ptrdiff_t) i11 * (ptrdiff_t) nb02 + (ptrdiff_t) i12 * (ptrdiff_t) nb03;
        dst_t * dst_row       = dst + i10 * s1 + i11 * s2 + i12 * s3;

        for (int64_t col = 0; col < ne00; ++col) {
            dst_row[col] = cast_from_f32<dst_t>((float) ((const src_t *) src0_row)[col]);
        }
    });
}

template <typename dst_t>
static void get_rows_q8_0(sycl::queue & q, const void * src0, const int32_t * rows, dst_t * dst,
                          int64_t ne00, int64_t ne10, int64_t ne11, int64_t ne12,
                          size_t nb01, size_t nb02, size_t nb03,
                          size_t nb1, size_t nb2, size_t nb3) {
    const size_t s1    = nb1 / sizeof(dst_t);
    const size_t s2    = nb2 / sizeof(dst_t);
    const size_t s3    = nb3 / sizeof(dst_t);
    const int64_t nblk = ne00 / QK8_0;

    q.parallel_for(sycl::range<3>((size_t) ne12, (size_t) ne11, (size_t) ne10), [=](sycl::item<3> item) {
        const int64_t i12 = item.get_id(0);
        const int64_t i11 = item.get_id(1);
        const int64_t i10 = item.get_id(2);
        const int64_t row_idx = i10 + ne10 * (i11 + ne11 * i12);
        const int32_t src_row = rows[row_idx];
        const block_q8_0 * src0_row = (const block_q8_0 *) ((const char *) src0 +
            (ptrdiff_t) src_row * (ptrdiff_t) nb01 + (ptrdiff_t) i11 * (ptrdiff_t) nb02 + (ptrdiff_t) i12 * (ptrdiff_t) nb03);
        dst_t * dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;

        for (int64_t ib = 0; ib < nblk; ++ib) {
            const float d = (float) src0_row[ib].d;
            for (int64_t iqs = 0; iqs < QK8_0; ++iqs) {
                dst_row[ib * QK8_0 + iqs] = cast_from_f32<dst_t>(d * (float) src0_row[ib].qs[iqs]);
            }
        }
    });
}

template <typename dst_t>
static void get_rows_q4_0(sycl::queue & q, const void * src0, const int32_t * rows, dst_t * dst,
                          int64_t ne00, int64_t ne10, int64_t ne11, int64_t ne12,
                          size_t nb01, size_t nb02, size_t nb03,
                          size_t nb1, size_t nb2, size_t nb3) {
    const size_t s1    = nb1 / sizeof(dst_t);
    const size_t s2    = nb2 / sizeof(dst_t);
    const size_t s3    = nb3 / sizeof(dst_t);
    const int64_t nblk = ne00 / QK4_0;

    q.parallel_for(sycl::range<3>((size_t) ne12, (size_t) ne11, (size_t) ne10), [=](sycl::item<3> item) {
        const int64_t i12 = item.get_id(0);
        const int64_t i11 = item.get_id(1);
        const int64_t i10 = item.get_id(2);
        const int64_t row_idx = i10 + ne10 * (i11 + ne11 * i12);
        const int32_t src_row = rows[row_idx];
        const block_q4_0 * src0_row = (const block_q4_0 *) ((const char *) src0 +
            (ptrdiff_t) src_row * (ptrdiff_t) nb01 + (ptrdiff_t) i11 * (ptrdiff_t) nb02 + (ptrdiff_t) i12 * (ptrdiff_t) nb03);
        dst_t * dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;

        for (int64_t ib = 0; ib < nblk; ++ib) {
            const float d = (float) src0_row[ib].d;
            for (int64_t iqs = 0; iqs < QK4_0; ++iqs) {
                const uint8_t q = src0_row[ib].qs[iqs / 2];
                const int vq = (iqs & 1) ? (q >> 4) : (q & 0x0F);
                dst_row[ib * QK4_0 + iqs] = cast_from_f32<dst_t>(((float) vq - 8.0f) * d);
            }
        }
    });
}

template <typename dst_t>
static void get_rows_q4_1(sycl::queue & q, const void * src0, const int32_t * rows, dst_t * dst,
                          int64_t ne00, int64_t ne10, int64_t ne11, int64_t ne12,
                          size_t nb01, size_t nb02, size_t nb03,
                          size_t nb1, size_t nb2, size_t nb3) {
    const size_t s1    = nb1 / sizeof(dst_t);
    const size_t s2    = nb2 / sizeof(dst_t);
    const size_t s3    = nb3 / sizeof(dst_t);
    const int64_t nblk = ne00 / QK4_1;

    q.parallel_for(sycl::range<3>((size_t) ne12, (size_t) ne11, (size_t) ne10), [=](sycl::item<3> item) {
        const int64_t i12 = item.get_id(0);
        const int64_t i11 = item.get_id(1);
        const int64_t i10 = item.get_id(2);
        const int64_t row_idx = i10 + ne10 * (i11 + ne11 * i12);
        const int32_t src_row = rows[row_idx];
        const block_q4_1 * src0_row = (const block_q4_1 *) ((const char *) src0 +
            (ptrdiff_t) src_row * (ptrdiff_t) nb01 + (ptrdiff_t) i11 * (ptrdiff_t) nb02 + (ptrdiff_t) i12 * (ptrdiff_t) nb03);
        dst_t * dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;

        for (int64_t ib = 0; ib < nblk; ++ib) {
            const float d = (float) src0_row[ib].GGML_COMMON_AGGR_S.d;
            const float m = (float) src0_row[ib].GGML_COMMON_AGGR_S.m;
            for (int64_t iqs = 0; iqs < QK4_1; ++iqs) {
                const uint8_t q = src0_row[ib].qs[iqs / 2];
                const int vq = (iqs & 1) ? (q >> 4) : (q & 0x0F);
                dst_row[ib * QK4_1 + iqs] = cast_from_f32<dst_t>((float) vq * d + m);
            }
        }
    });
}

template <typename dst_t>
static void dispatch_src0_type(sycl::queue & q, const ggml_tensor * src0, const int32_t * rows, ggml_tensor * dst) {
    dst_t * dst_d          = (dst_t *) dst->data;

    switch (src0->type) {
        case GGML_TYPE_F32:
            get_rows_dense<float, dst_t>(q, (const float *) src0->data, rows, dst_d,
                src0->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                src0->nb[1], src0->nb[2], src0->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        case GGML_TYPE_F16:
            get_rows_dense<sycl::half, dst_t>(q, (const sycl::half *) src0->data, rows, dst_d,
                src0->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                src0->nb[1], src0->nb[2], src0->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_q8_0<dst_t>(q, src0->data, rows, dst_d,
                src0->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                src0->nb[1], src0->nb[2], src0->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_q4_0<dst_t>(q, src0->data, rows, dst_d,
                src0->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                src0->nb[1], src0->nb[2], src0->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_q4_1<dst_t>(q, src0->data, rows, dst_d,
                src0->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                src0->nb[1], src0->nb[2], src0->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        default:
            GGML_ABORT("%s: unsupported src0 type for SYCL get_rows: %s", __func__, ggml_type_name(src0->type));
    }
}

} // namespace

void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->ne[3] == 1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    sycl::queue & q = ctx.stream();
    const int64_t nr = src1->ne[0] * src1->ne[1] * src1->ne[2];
    int32_t * rows = sycl::malloc_shared<int32_t>(nr, q);

    GGML_ASSERT(rows != nullptr);

    q.memcpy(rows, src1->data, (size_t) nr * sizeof(int32_t)).wait();

    for (int64_t i = 0; i < nr; ++i) {
        GGML_ASSERT(rows[i] >= 0 && rows[i] < src0->ne[1]);
    }

    switch (dst->type) {
        case GGML_TYPE_F32:
            dispatch_src0_type<float>(q, src0, rows, dst);
            break;
        case GGML_TYPE_F16:
            dispatch_src0_type<sycl::half>(q, src0, rows, dst);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type for SYCL get_rows: %s", __func__, ggml_type_name(dst->type));
    }

    q.wait_and_throw();
    sycl::free(rows, q);
}

bool ggml_sycl_supports_get_rows(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }

    if (src1->type != GGML_TYPE_I32 || src1->ne[3] != 1) {
        return false;
    }

    if (op->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous(src1)) {
        return false;
    }

    if (src0->ne[2] != src1->ne[1] || src0->ne[3] != src1->ne[2]) {
        return false;
    }

    if (src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != ggml_type_size(src1->type) || op->nb[0] != ggml_type_size(op->type)) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return true;
        case GGML_TYPE_Q8_0:
            return src0->ne[0] % QK8_0 == 0;
        default:
            return false;
    }
}
