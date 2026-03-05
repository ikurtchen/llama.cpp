#include "getrows.hpp"
#include "dequantize.hpp"

#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #include <sycl/ext/oneapi/bfloat16.hpp>
#endif
using sycl_bf16 = sycl::ext::oneapi::bfloat16;

// Cast helper (equivalent to CUDA's ggml_cuda_cast)
template<typename dst_t, typename src_t>
static inline dst_t sycl_cast(src_t x) {
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return x;
    } else if constexpr (std::is_same_v<dst_t, int32_t>) {
        return static_cast<int32_t>(x);
    } else {
        return static_cast<dst_t>(float(x));
    }
}

// Dequantize traits for compile-time dispatch (avoids function pointers in device code)
template<ggml_type type> struct dequant_traits;

template<> struct dequant_traits<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0;
    static constexpr int qr = QR4_0;
    static void dequantize(const void * vx, int64_t ib, int iqs, dfloat2 & v) {
        dequantize_q4_0(vx, ib, iqs, v);
    }
};

template<> struct dequant_traits<GGML_TYPE_Q4_1> {
    static constexpr int qk = QK4_1;
    static constexpr int qr = QR4_1;
    static void dequantize(const void * vx, int64_t ib, int iqs, dfloat2 & v) {
        dequantize_q4_1(vx, ib, iqs, v);
    }
};

template<> struct dequant_traits<GGML_TYPE_Q5_0> {
    static constexpr int qk = QK5_0;
    static constexpr int qr = QR5_0;
    static void dequantize(const void * vx, int64_t ib, int iqs, dfloat2 & v) {
        dequantize_q5_0(vx, ib, iqs, v);
    }
};

template<> struct dequant_traits<GGML_TYPE_Q5_1> {
    static constexpr int qk = QK5_1;
    static constexpr int qr = QR5_1;
    static void dequantize(const void * vx, int64_t ib, int iqs, dfloat2 & v) {
        dequantize_q5_1(vx, ib, iqs, v);
    }
};

template<> struct dequant_traits<GGML_TYPE_Q8_0> {
    static constexpr int qk = QK8_0;
    static constexpr int qr = QR8_0;
    static void dequantize(const void * vx, int64_t ib, int iqs, dfloat2 & v) {
        dequantize_q8_0(vx, ib, iqs, v);
    }
};

// ============================================================
// Kernel: k_get_rows_quant — for quantized source types
// ============================================================
// nd_range<3> mapping:
//   dim0 (group(0)) → i10 (row selector in src1), count = ne10
//   dim1 (group(1)) → column blocks for i00 partitioning, count = min(block_num_y, 65535)
//   dim2 (group(2)) → z-dimension for ne11*ne12 (capped), local_range = BLOCK_SIZE
template<ggml_type qtype, typename dst_t>
static void k_get_rows_quant(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        int64_t ne00, int64_t ne11, int64_t ne12,
        size_t s1, size_t s2, size_t s3,
        size_t nb01, size_t nb02, size_t nb03,
        size_t s10, size_t s11, size_t s12,
        int gridDim_y, int gridDim_z,
        const sycl::nd_item<3> & item) {

    using traits = dequant_traits<qtype>;
    constexpr int qk = traits::qk;
    constexpr int qr = traits::qr;

    const int thread    = item.get_local_id(2);
    const int block_sz  = item.get_local_range(2);

    for (int64_t z = item.get_group(2); z < ne11 * ne12; z += gridDim_z) {
        for (int64_t i00 = 2 * (item.get_group(1) * block_sz + thread); i00 < ne00;
             i00 += 2 * gridDim_y * block_sz) {
            const int i10 = item.get_group(0);
            const int i11 = z / ne12;
            const int i12 = z % ne12;

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t      * dst_row  = dst + i10*s1 + i11*s2 + i12*s3;
            const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

            const int ib       = i00 / qk;
            const int iqs      = (i00 % qk) / qr;
            const int iybs     = i00 - i00 % qk;
            const int y_offset = qr == 1 ? 1 : qk / 2;

            dfloat2 v;
            traits::dequantize(src0_row, ib, iqs, v);

            dst_row[iybs + iqs + 0]        = sycl_cast<dst_t>(float(v.x()));
            dst_row[iybs + iqs + y_offset]  = sycl_cast<dst_t>(float(v.y()));
        }
    }
}

// ============================================================
// Kernel: k_get_rows_float — for float source types
// ============================================================
template<typename src0_t, typename dst_t>
static void k_get_rows_float(
        const src0_t * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        int64_t ne00, int64_t ne11, int64_t ne12,
        size_t s1, size_t s2, size_t s3,
        size_t nb01, size_t nb02, size_t nb03,
        size_t s10, size_t s11, size_t s12,
        int gridDim_y, int gridDim_z,
        const sycl::nd_item<3> & item) {

    const int thread    = item.get_local_id(2);
    const int block_sz  = item.get_local_range(2);

    for (int64_t z = item.get_group(2); z < ne11 * ne12; z += gridDim_z) {
        for (int64_t i00 = item.get_group(1) * block_sz + thread; i00 < ne00;
             i00 += gridDim_y * block_sz) {
            const int i10 = item.get_group(0);
            const int i11 = z / ne12;
            const int i12 = z % ne12;

            if (i00 >= ne00) {
                return;
            }

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t        * dst_row  = dst + i10*s1 + i11*s2 + i12*s3;
            const src0_t * src0_row = (const src0_t *)((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

            dst_row[i00] = sycl_cast<dst_t>(src0_row[i00]);
        }
    }
}

// ============================================================
// Kernel: k_get_rows_back_float — backward pass (scatter-add by index)
// ============================================================
// nd_range<3> mapping:
//   dim0 = 1 (unused)
//   dim1 (group(1)) → dst rows (ne1)
//   dim2 (group(2)*local(2) + local_id(2)) → columns (ne00)
template<typename grad_t, typename dst_t>
static void k_get_rows_back_float(
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows,
        dst_t * __restrict__ dst,
        int64_t ncols, int64_t nrows_grad,
        const sycl::nd_item<3> & item) {

    const int col = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
    if (col >= ncols) {
        return;
    }

    const int dst_row = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

    float sum = 0.0f;
    for (int64_t i = 0; i < nrows_grad; ++i) {
        if (rows[i] != dst_row) {
            continue;
        }
        sum += float(grad[i * ncols + col]);
    }

    dst[dst_row * ncols + col] = sycl_cast<dst_t>(sum);
}

// ============================================================
// Launch helper: quantized get_rows
// ============================================================
template<ggml_type qtype, typename dst_t>
static void get_rows_sycl_q(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        sycl::queue & stream) {

    constexpr int block_size = SYCL_GET_ROWS_BLOCK_SIZE;
    const int block_num_y = (ne00 + 2*block_size - 1) / (2*block_size);

    const int gridDim_y = std::min(block_num_y, (int)UINT16_MAX);
    const int gridDim_z = (int)std::min((int64_t)UINT16_MAX, ne11 * ne12);

    const size_t s1  = nb1  / sizeof(dst_t);
    const size_t s2  = nb2  / sizeof(dst_t);
    const size_t s3  = nb3  / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    GGML_ASSERT(ne00 % 2 == 0);

    // global_range must be a multiple of local_range in each dimension
    sycl::range<3> global((size_t)ne10, (size_t)gridDim_y, (size_t)gridDim_z * block_size);
    sycl::range<3> local(1, 1, block_size);

    stream.parallel_for(sycl::nd_range<3>(global, local),
        [=](sycl::nd_item<3> item) {
            k_get_rows_quant<qtype, dst_t>(
                src0_d, src1_d, dst_d,
                ne00, ne11, ne12, s1, s2, s3,
                nb01, nb02, nb03, s10, s11, s12,
                gridDim_y, gridDim_z, item);
        });
}

// ============================================================
// Launch helper: float get_rows
// ============================================================
template<typename src0_t, typename dst_t>
static void get_rows_sycl_float(
        const src0_t * src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        sycl::queue & stream) {

    constexpr int block_size = SYCL_GET_ROWS_BLOCK_SIZE;
    const int block_num_y = (ne00 + block_size - 1) / block_size;

    const int gridDim_y = std::min(block_num_y, (int)UINT16_MAX);
    const int gridDim_z = (int)std::min((int64_t)UINT16_MAX, ne11 * ne12);

    const size_t s1  = nb1  / sizeof(dst_t);
    const size_t s2  = nb2  / sizeof(dst_t);
    const size_t s3  = nb3  / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    sycl::range<3> global((size_t)ne10, (size_t)gridDim_y, (size_t)gridDim_z * block_size);
    sycl::range<3> local(1, 1, block_size);

    stream.parallel_for(sycl::nd_range<3>(global, local),
        [=](sycl::nd_item<3> item) {
            k_get_rows_float<src0_t, dst_t>(
                src0_d, src1_d, dst_d,
                ne00, ne11, ne12, s1, s2, s3,
                nb01, nb02, nb03, s10, s11, s12,
                gridDim_y, gridDim_z, item);
        });
}

// ============================================================
// Type dispatch — switch on src0 type
// ============================================================
template<typename dst_t>
static void get_rows_sycl_switch_src0(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        sycl::queue & stream) {
    switch (src0_type) {
        case GGML_TYPE_F16:
            get_rows_sycl_float((const sycl::half *)src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_sycl_float((const float *)src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            get_rows_sycl_float((const int32_t *)src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            get_rows_sycl_float((const sycl_bf16 *)src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_sycl_q<GGML_TYPE_Q4_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_sycl_q<GGML_TYPE_Q4_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_sycl_q<GGML_TYPE_Q5_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_sycl_q<GGML_TYPE_Q5_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_sycl_q<GGML_TYPE_Q8_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0_type));
            break;
    }
}

// ============================================================
// Top-level dispatch — switch on dst type
// ============================================================
static void get_rows_sycl(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        sycl::queue & stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:
            get_rows_sycl_switch_src0(src0_d, src0_type, src1_d, (float *)dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            get_rows_sycl_switch_src0(src0_d, src0_type, src1_d, (int32_t *)dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F16:
            get_rows_sycl_switch_src0(src0_d, src0_type, src1_d, (sycl::half *)dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            get_rows_sycl_switch_src0(src0_d, src0_type, src1_d, (sycl_bf16 *)dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(dst_type));
            break;
    }
}

// ============================================================
// Public API: ggml_sycl_op_get_rows
// ============================================================
void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    sycl::queue & stream = *(ctx.stream());

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ne13 == 1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    get_rows_sycl(src0->data, src0->type, (const int32_t *)src1->data, dst->data, dst->type,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

// ============================================================
// Public API: ggml_sycl_op_get_rows_back
// ============================================================
void ggml_sycl_op_get_rows_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // indices (src1 in forward pass)

    GGML_TENSOR_BINARY_OP_LOCALS

    const float   * src0_d = (const float   *)src0->data;
    const int32_t * src1_d = (const int32_t *)src1->data;
    float         * dst_d  = (float         *)dst->data;

    sycl::queue & stream = *(ctx.stream());

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);

    constexpr int block_size = SYCL_GET_ROWS_BLOCK_SIZE;
    const int block_num_x = (ne00 + block_size - 1) / block_size;

    // Matches CUDA layout: grid = (block_num_x, ne1, 1), block = (block_size, 1, 1)
    // SYCL mapping: dim2 = columns (threaded), dim1 = dst rows, dim0 = 1
    sycl::range<3> global(1, (size_t)ne1, (size_t)block_num_x * block_size);
    sycl::range<3> local(1, 1, block_size);

    stream.parallel_for(sycl::nd_range<3>(global, local),
        [=](sycl::nd_item<3> item) {
            k_get_rows_back_float<float, float>(
                src0_d, src1_d, dst_d, ne00, ne10, item);
        });
}
