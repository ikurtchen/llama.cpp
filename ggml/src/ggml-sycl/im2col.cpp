#include "im2col.hpp"

#include <sycl/sycl.hpp>
#include <type_traits>

namespace {
template <typename T>
inline T im2col_from_float(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else {
        return (sycl::half) value;
    }
}

template <typename T>
void im2col_2d_dispatch(
        sycl::queue & q,
        const float * src,
        T * dst,
        int64_t ic,
        int64_t iw,
        int64_t ih,
        int64_t ow,
        int64_t oh,
        int64_t kw,
        int64_t kh,
        int64_t n,
        int64_t ic_ih_iw,
        int64_t ih_iw,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1) {
    const int64_t total = n * oh * ow * ic * kh * kw;
    const int64_t block = 256;
    const int64_t nblk = (total + block - 1) / block;
    const int64_t kh_kw = kh * kw;
    const int64_t ic_kh_kw = ic * kh_kw;

    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
            [=](sycl::nd_item<1> item) {
                const int64_t idx = item.get_global_id(0);
                if (idx >= total) {
                    return;
                }

                int64_t rem = idx;
                const int64_t i0 = rem % ic_kh_kw;
                rem /= ic_kh_kw;
                const int64_t iow = rem % ow;
                rem /= ow;
                const int64_t ioh = rem % oh;
                const int64_t in = rem / oh;

                const int64_t iic = i0 / kh_kw;
                const int64_t ik = i0 - iic * kh_kw;
                const int64_t ikh = ik / kw;
                const int64_t ikw = ik - ikh * kw;

                const int64_t iiw = iow * s0 + ikw * d0 - p0;
                const int64_t iih = ioh * s1 + ikh * d1 - p1;

                float value = 0.0f;
                if (0 <= iiw && iiw < iw && 0 <= iih && iih < ih) {
                    const int64_t offset_src = iic * ic_ih_iw + in * ih_iw + iih * iw + iiw;
                    value = src[offset_src];
                }

                dst[idx] = im2col_from_float<T>(value);
            });
}

template <typename T>
void im2col_3d_dispatch(
        sycl::queue & q,
        const float * src,
        T * dst,
        int64_t n,
        int64_t ic,
        int64_t id,
        int64_t ih,
        int64_t iw,
        int64_t kd,
        int64_t kh,
        int64_t kw,
        int64_t od,
        int64_t oh,
        int64_t ow,
        int64_t stride_q,
        int64_t stride_z,
        int64_t stride_y,
        int64_t stride_x,
        int s0,
        int s1,
        int s2,
        int p0,
        int p1,
        int p2,
        int d0,
        int d1,
        int d2) {
    const int64_t kd_kh_kw = kd * kh * kw;
    const int64_t ic_kd_kh_kw = ic * kd_kh_kw;
    const int64_t total = n * od * oh * ow * ic_kd_kh_kw;
    const int64_t block = 256;
    const int64_t nblk = (total + block - 1) / block;
    const int64_t kh_kw = kh * kw;

    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
            [=](sycl::nd_item<1> item) {
                const int64_t idx = item.get_global_id(0);
                if (idx >= total) {
                    return;
                }

                int64_t rem = idx;
                const int64_t i0 = rem % ic_kd_kh_kw;
                rem /= ic_kd_kh_kw;
                const int64_t iow = rem % ow;
                rem /= ow;
                const int64_t ioh = rem % oh;
                rem /= oh;
                const int64_t iodn = rem;
                const int64_t in = iodn / od;
                const int64_t iod = iodn - in * od;

                const int64_t iic = i0 / kd_kh_kw;
                const int64_t ikd_rem = i0 - iic * kd_kh_kw;
                const int64_t ikd = ikd_rem / kh_kw;
                const int64_t ikh_rem = ikd_rem - ikd * kh_kw;
                const int64_t ikh = ikh_rem / kw;
                const int64_t ikw = ikh_rem - ikh * kw;

                const int64_t iiw = iow * s0 + ikw * d0 - p0;
                const int64_t iih = ioh * s1 + ikh * d1 - p1;
                const int64_t iid = iod * s2 + ikd * d2 - p2;

                float value = 0.0f;
                if (0 <= iiw && iiw < iw && 0 <= iih && iih < ih && 0 <= iid && iid < id) {
                    const int64_t offset_src = ((in * ic + iic) * stride_q) + iid * stride_z + iih * stride_y + iiw * stride_x;
                    value = src[offset_src];
                }

                dst[idx] = im2col_from_float<T>(value);
            });
}
}

void ggml_sycl_op_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
    const int32_t s1 = ggml_get_op_params_i32(dst, 1);
    const int32_t p0 = ggml_get_op_params_i32(dst, 2);
    const int32_t p1 = ggml_get_op_params_i32(dst, 3);
    const int32_t d0 = ggml_get_op_params_i32(dst, 4);
    const int32_t d1 = ggml_get_op_params_i32(dst, 5);
    const bool is_2d = ggml_get_op_params_i32(dst, 6) == 1;

    const int64_t ic = src1->ne[is_2d ? 2 : 1];
    const int64_t ih = is_2d ? src1->ne[1] : 1;
    const int64_t iw = src1->ne[0];

    const int64_t kh = is_2d ? src0->ne[1] : 1;
    const int64_t kw = src0->ne[0];

    const int64_t oh = is_2d ? dst->ne[2] : 1;
    const int64_t ow = dst->ne[1];

    const int64_t ic_ih_iw = src1->nb[is_2d ? 2 : 1] / sizeof(float);
    const int64_t n = src1->ne[is_2d ? 3 : 2];
    const int64_t ih_iw = src1->nb[is_2d ? 3 : 2] / sizeof(float);

    sycl::queue & q = ctx.stream();
    if (dst->type == GGML_TYPE_F32) {
        im2col_2d_dispatch(q, (const float *) src1->data, (float *) dst->data, ic, iw, ih, ow, oh, kw, kh, n,
                ic_ih_iw, ih_iw, s0, s1, p0, p1, d0, d1);
    } else {
        im2col_2d_dispatch(q, (const float *) src1->data, (sycl::half *) dst->data, ic, iw, ih, ow, oh, kw, kh, n,
                ic_ih_iw, ih_iw, s0, s1, p0, p1, d0, d1);
    }
}

void ggml_sycl_op_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
    const int32_t s1 = ggml_get_op_params_i32(dst, 1);
    const int32_t s2 = ggml_get_op_params_i32(dst, 2);
    const int32_t p0 = ggml_get_op_params_i32(dst, 3);
    const int32_t p1 = ggml_get_op_params_i32(dst, 4);
    const int32_t p2 = ggml_get_op_params_i32(dst, 5);
    const int32_t d0 = ggml_get_op_params_i32(dst, 6);
    const int32_t d1 = ggml_get_op_params_i32(dst, 7);
    const int32_t d2 = ggml_get_op_params_i32(dst, 8);
    const int32_t ic = ggml_get_op_params_i32(dst, 9);

    const int64_t n = ne13 / ic;
    const int64_t id = ne12;
    const int64_t ih = ne11;
    const int64_t iw = ne10;

    const int64_t kd = ne02;
    const int64_t kh = ne01;
    const int64_t kw = ne00;

    const int64_t od = ne3 / n;
    const int64_t oh = ne2;
    const int64_t ow = ne1;

    const int64_t stride_x = src1->nb[0] / sizeof(float);
    const int64_t stride_y = src1->nb[1] / sizeof(float);
    const int64_t stride_z = src1->nb[2] / sizeof(float);
    const int64_t stride_q = src1->nb[3] / sizeof(float);

    sycl::queue & q = ctx.stream();
    if (dst->type == GGML_TYPE_F32) {
        im2col_3d_dispatch(q, (const float *) src1->data, (float *) dst->data, n, ic, id, ih, iw, kd, kh, kw, od, oh, ow,
                stride_q, stride_z, stride_y, stride_x, s0, s1, s2, p0, p1, p2, d0, d1, d2);
    } else {
        im2col_3d_dispatch(q, (const float *) src1->data, (sycl::half *) dst->data, n, ic, id, ih, iw, kd, kh, kw, od, oh, ow,
                stride_q, stride_z, stride_y, stride_x, s0, s1, s2, p0, p1, p2, d0, d1, d2);
    }
}

bool ggml_sycl_supports_im2col(const ggml_tensor * op) {
    return ggml_is_contiguous(op->src[0]) &&
           ggml_is_contiguous(op->src[1]) &&
           op->src[1]->type == GGML_TYPE_F32 &&
           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
}

bool ggml_sycl_supports_im2col_3d(const ggml_tensor * op) {
    return ggml_is_contiguous(op->src[0]) &&
           ggml_is_contiguous(op->src[1]) &&
           op->src[1]->type == GGML_TYPE_F32 &&
           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
}
