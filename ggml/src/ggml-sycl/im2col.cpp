#include "im2col.hpp"

#define MAX_GRIDDIM_Z 65535

template <typename T>
static void im2col_sycl(
        const float * x, T * dst,
        int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
        int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
        int s0, int s1, int p0, int p1, int d0, int d1, dpct::queue_ptr stream) {
    const int64_t IC_KH_KW = IC * KH * KW;
    const int64_t KH_KW = KH * KW;
    const int64_t N_OH = N * OH;
    const int64_t num_blocks = (IC_KH_KW + SYCL_IM2COL_BLOCK_SIZE - 1) / SYCL_IM2COL_BLOCK_SIZE;
    const int64_t local_size = std::min(IC_KH_KW, (int64_t)SYCL_IM2COL_BLOCK_SIZE);
    const int64_t grid_z = std::min(N_OH, (int64_t)MAX_GRIDDIM_Z);

    sycl::range<3> global_range(grid_z, OW, num_blocks * local_size);
    sycl::range<3> local_range(1, 1, local_size);

    stream->parallel_for(
        sycl::nd_range<3>(global_range, local_range),
        [=](sycl::nd_item<3> item) {
            const int64_t i = item.get_global_id(2);
            if (i >= IC_KH_KW) {
                return;
            }

            const int64_t iic = i / KH_KW;
            const int64_t rem = i - iic * KH_KW;
            const int64_t ikh = rem / KW;
            const int64_t ikw = rem - ikh * KW;

            const int64_t iow = item.get_group(1);
            for (int64_t iz = item.get_group(0); iz < N_OH; iz += MAX_GRIDDIM_Z) {
                const int64_t in  = iz / OH;
                const int64_t ioh = iz - in * OH;

                const int64_t iiw = iow * s0 + ikw * d0 - p0;
                const int64_t iih = ioh * s1 + ikh * d1 - p1;

                const int64_t offset_dst =
                    ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

                if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                    dst[offset_dst] = static_cast<T>(0.0f);
                } else {
                    const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
                    dst[offset_dst] = static_cast<T>(x[offset_src + iih * IW + iiw]);
                }
            }
        });
}

void ggml_sycl_op_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[5];

    const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

    const int64_t IC = src1->ne[is_2D ? 2 : 1];
    const int64_t IH = is_2D ? src1->ne[1] : 1;
    const int64_t IW =         src1->ne[0];

    const int64_t KH = is_2D ? src0->ne[1] : 1;
    const int64_t KW =         src0->ne[0];

    const int64_t OH = is_2D ? dst->ne[2] : 1;
    const int64_t OW =         dst->ne[1];

    const int64_t IC_IH_IW = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    const int64_t N        = src1->ne[is_2D ? 3 : 2];
    const int64_t IH_IW    = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    if (dst->type == GGML_TYPE_F16) {
        im2col_sycl<sycl::half>(src1_d, (sycl::half *)dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    } else {
        im2col_sycl<float>(src1_d, (float *)dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
static void im2col_3d_sycl(
        const float * src, T * dst,
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
        int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
        int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
        int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2,
        dpct::queue_ptr stream) {
    const int64_t KD_KH_KW = KD * KH * KW;
    const int64_t KH_KW = KH * KW;
    const int64_t IC_KD_KH_KW = IC * KD_KH_KW;
    const int64_t N_OD_OH = N * OD * OH;
    const int64_t OD_OH = OD * OH;
    const int64_t OD_OH_OW_IC_KD_KH_KW = OD * OH * OW * IC_KD_KH_KW;
    const int64_t OH_OW_IC_KD_KH_KW = OH * OW * IC_KD_KH_KW;
    const int64_t OW_IC_KD_KH_KW = OW * IC_KD_KH_KW;
    const int64_t num_blocks = (IC_KD_KH_KW + SYCL_IM2COL_BLOCK_SIZE - 1) / SYCL_IM2COL_BLOCK_SIZE;
    const int64_t local_size = std::min(IC_KD_KH_KW, (int64_t)SYCL_IM2COL_BLOCK_SIZE);
    const int64_t grid_z = std::min(N_OD_OH, (int64_t)MAX_GRIDDIM_Z);

    sycl::range<3> global_range(grid_z, OW, num_blocks * local_size);
    sycl::range<3> local_range(1, 1, local_size);

    // Suppress unused variable warnings (matching CUDA kernel)
    (void)OC;

    stream->parallel_for(
        sycl::nd_range<3>(global_range, local_range),
        [=](sycl::nd_item<3> item) {
            const int64_t i = item.get_global_id(2);
            if (i >= IC_KD_KH_KW) {
                return;
            }

            const int64_t iic = i / KD_KH_KW;
            const int64_t ikd = (i - iic * KD_KH_KW) / KH_KW;
            const int64_t ikh = (i - iic * KD_KH_KW - ikd * KH_KW) / KW;
            const int64_t ikw = i % KW;

            const int64_t iow = item.get_group(1);
            for (int64_t iz = item.get_group(0); iz < N_OD_OH; iz += MAX_GRIDDIM_Z) {
                const int64_t in  = iz / OD_OH;
                const int64_t iod = (iz - in * OD_OH) / OH;
                const int64_t ioh = iz % OH;

                const int64_t iiw = iow * s0 + ikw * d0 - p0;
                const int64_t iih = ioh * s1 + ikh * d1 - p1;
                const int64_t iid = iod * s2 + ikd * d2 - p2;

                const int64_t offset_dst = in * OD_OH_OW_IC_KD_KH_KW + iod * OH_OW_IC_KD_KH_KW
                    + ioh * OW_IC_KD_KH_KW + iow * IC_KD_KH_KW + iic * KD_KH_KW
                    + ikd * KH_KW + ikh * KW + ikw;

                if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                    dst[offset_dst] = static_cast<T>(0.0f);
                } else {
                    const int64_t offset_src = ((in * IC + iic) * stride_q) + (iid * stride_z)
                        + (iih * stride_y) + (iiw * stride_x);
                    dst[offset_dst] = static_cast<T>(src[offset_src]);
                }
            }
        });
}

void ggml_sycl_op_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t s2 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[3];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[4];
    const int32_t p2 = ((const int32_t *)(dst->op_params))[5];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[6];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[7];
    const int32_t d2 = ((const int32_t *)(dst->op_params))[8];
    const int32_t IC = ((const int32_t *)(dst->op_params))[9];

    const int64_t N  = ne13 / IC;
    const int64_t ID = ne12;
    const int64_t IH = ne11;
    const int64_t IW = ne10;

    const int64_t OC = ne03 / IC;
    const int64_t KD = ne02;
    const int64_t KH = ne01;
    const int64_t KW = ne00;

    const int64_t OD = ne3 / N;
    const int64_t OH = ne2;
    const int64_t OW = ne1;

    const size_t  es       = ggml_element_size(src1);
    const int64_t stride_x = src1->nb[0] / es;
    const int64_t stride_y = src1->nb[1] / es;
    const int64_t stride_z = src1->nb[2] / es;
    const int64_t stride_q = src1->nb[3] / es;

    if (dst->type == GGML_TYPE_F16) {
        im2col_3d_sycl<sycl::half>(src1_d, (sycl::half *)dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                                    stride_q, stride_z, stride_y, stride_x,
                                    s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    } else {
        im2col_3d_sycl<float>(src1_d, (float *)dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                              stride_q, stride_z, stride_y, stride_x,
                              s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    }
}
