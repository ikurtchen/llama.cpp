#include "cpy.hpp"

#include <sycl/sycl.hpp>
#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #include <sycl/ext/oneapi/bfloat16.hpp>
#endif
#include <cstring>

using sycl_bf16 = sycl::ext::oneapi::bfloat16;

// ============================================================
// Kernel: cpy_scalar — generic non-contiguous element copy
// ============================================================
// Maps 1D flat index i → 4D coordinates → byte offsets for both src and dst
template<typename src_t, typename dst_t>
static void k_cpy_scalar(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    const sycl::nd_item<1> & item) {

    const int64_t i = (int64_t)item.get_global_id(0);
    if (i >= ne) return;

    // 4D index decomposition for source
    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02) / (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02 - i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;

    // 4D index decomposition for destination
    const int64_t i13 = i / (ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    *(dst_t *)(cdst + dst_offset) = (dst_t)(*(const src_t *)(cx + x_offset));
}

// ============================================================
// Kernel: cpy_scalar_contiguous — optimized same-layout copy
// ============================================================
template<typename src_t, typename dst_t>
static void k_cpy_scalar_contiguous(
    const char * cx, char * cdst, const int64_t ne,
    const sycl::nd_item<1> & item) {

    const int64_t i = (int64_t)item.get_global_id(0);
    if (i >= ne) return;

    const src_t * x = (const src_t *) cx;
    dst_t * dst = (dst_t *) cdst;
    dst[i] = (dst_t)(x[i]);
}

// ============================================================
// Launch helpers
// ============================================================
// SYCL_CPY_BLOCK_SIZE is defined in presets.hpp

template<typename src_t, typename dst_t>
static void launch_cpy_scalar_contiguous(
    const char * cx, char * cdst, const int64_t ne, sycl::queue & stream) {

    const int64_t num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_CPY_BLOCK_SIZE, SYCL_CPY_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            k_cpy_scalar_contiguous<src_t, dst_t>(cx, cdst, ne, item);
        });
}

template<typename src_t, typename dst_t>
static void launch_cpy_scalar(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    sycl::queue & stream) {

    const int64_t num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_CPY_BLOCK_SIZE, SYCL_CPY_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            k_cpy_scalar<src_t, dst_t>(cx, cdst, ne,
                ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                ne10, ne11, ne12, nb10, nb11, nb12, nb13, item);
        });
}

// ============================================================
// ggml_sycl_cpy — main dispatch (mirrors ggml_cuda_cpy)
// ============================================================
void ggml_sycl_cpy(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1) {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    sycl::queue & stream = *ctx.stream();

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;

    const bool contiguous_srcs = ggml_is_contiguous(src0) && ggml_is_contiguous(src1);

    // Fast path: same type + both contiguous → memcpy
    if (src0->type == src1->type && contiguous_srcs) {
        GGML_ASSERT(ggml_nbytes(src0) == ggml_nbytes(src1));
        stream.memcpy(src1_ddc, src0_ddc, ggml_nbytes(src0));
        return;
    }

    // f32 → f32
    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<float, float>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<float, float>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // f32 → f16
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<float, sycl::half>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<float, sycl::half>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // f16 → f32
    else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<sycl::half, float>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<sycl::half, float>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // f16 → f16
    else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<sycl::half, sycl::half>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<sycl::half, sycl::half>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // f32 → i32
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<float, int32_t>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<float, int32_t>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // i32 → f32
    else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<int32_t, float>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<int32_t, float>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // i32 → i32 (non-contiguous; contiguous already handled by memcpy above)
    else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        launch_cpy_scalar<int32_t, int32_t>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
    }
    // bf16 → bf16 (non-contiguous; contiguous already handled by memcpy above)
    else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<sycl_bf16, sycl_bf16>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<sycl_bf16, sycl_bf16>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // bf16 → f32
    else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<sycl_bf16, float>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<sycl_bf16, float>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // f32 → bf16
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<float, sycl_bf16>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<float, sycl_bf16>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // bf16 → f16
    else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<sycl_bf16, sycl::half>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<sycl_bf16, sycl::half>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // f16 → bf16
    else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<sycl::half, sycl_bf16>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<sycl::half, sycl_bf16>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    else {
        GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__,
            ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

// ============================================================
// ggml_sycl_dup — DUP and CONT use dup (= cpy src0 → dst)
// ============================================================
void ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_sycl_cpy(ctx, src0, dst);
}
