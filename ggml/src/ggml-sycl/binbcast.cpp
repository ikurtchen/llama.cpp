#include "binbcast.hpp"

#include <algorithm>
#include <cstdint>

// ============================================================================
// Binary operation enum + dispatch
// ============================================================================

enum class bin_op_type { ADD, SUB, MUL, DIV, REPEAT };

template <bin_op_type op>
static inline float apply_bin_op(float a, float b) {
    if constexpr (op == bin_op_type::ADD)    return a + b;
    if constexpr (op == bin_op_type::SUB)    return a - b;
    if constexpr (op == bin_op_type::MUL)    return a * b;
    if constexpr (op == bin_op_type::DIV)    return a / b;
    if constexpr (op == bin_op_type::REPEAT) return b;
}

// ============================================================================
// 3D broadcast kernel
// Grid: (ceil(ne0/2, blockDim.x), ceil(ne1, blockDim.y), ceil(ne2*ne3, blockDim.z))
// Uses plain integer % instead of CUDA fastdiv.
// ============================================================================

template <bin_op_type op, typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast(const src0_t * src0,
                        const src1_t * src1,
                        dst_t *        dst,
                        const int      ne0,
                        const int      ne1,
                        const int      ne2,
                        const uint32_t ne3,
                        const uint32_t ne10,
                        const uint32_t ne11,
                        const uint32_t ne12,
                        const uint32_t ne13,
                      /*const int      s0,*/
                        const int      s1,
                        const int      s2,
                        const int      s3,
                        const int      s00,
                        const int      s01,
                        const int      s02,
                        const int      s03,
                        const int      s10,
                        const int      s11,
                        const int      s12,
                        const int      s13,
                        const sycl::nd_item<3> & item) {

    const uint32_t i0s = item.get_local_range(2) * item.get_group(2) + item.get_local_id(2);
    const uint32_t i1  = item.get_local_range(1) * item.get_group(1) + item.get_local_id(1);
    const uint32_t i23 = item.get_local_range(0) * item.get_group(0) + item.get_local_id(0);
    const uint32_t i2  = i23 / ne3;
    const uint32_t i3  = i23 % ne3;

    if (i0s >= (uint32_t)ne0 || i1 >= (uint32_t)ne1 || i2 >= (uint32_t)ne2 || i3 >= ne3) {
        return;
    }

    const uint32_t i11 = i1 % ne11;
    const uint32_t i12 = i2 % ne12;
    const uint32_t i13 = i3 % ne13;

    const size_t i_src0 = (size_t)i3*s03 + (size_t)i2*s02 + (size_t)i1*s01;
    const size_t i_src1 = (size_t)i13*s13 + (size_t)i12*s12 + (size_t)i11*s11;
    const size_t i_dst  = (size_t)i3*s3  + (size_t)i2*s2  + (size_t)i1*s1;

    const src0_t * src0_row = src0 ? (src0 + i_src0) : nullptr;
    dst_t * dst_row = dst + i_dst;

    const uint32_t grid_x = item.get_local_range(2) * item.get_group_range(2);
    for (int i0 = i0s; i0 < ne0; i0 += grid_x) {
        const uint32_t i10 = i0 % ne10;

        float result = src0_row ? (float)src0_row[i0*s00] : 0.0f;
        result = apply_bin_op<op>(result, (float)src1[i_src1 + i10*s10]);

        dst_row[i0] = (dst_t)result;
    }
}

// ============================================================================
// 1D unravel broadcast kernel (fallback when grid dims exceed 65535)
// ============================================================================

template <bin_op_type op, typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast_unravel(const src0_t * src0,
                                const src1_t * src1,
                                dst_t *        dst,
                                const uint32_t ne0,
                                const uint32_t ne1,
                                const uint32_t ne2,
                                const uint32_t ne3,
                                const uint32_t prod_012,
                                const uint32_t prod_01,
                                const uint32_t ne10,
                                const uint32_t ne11,
                                const uint32_t ne12,
                                const uint32_t ne13,
                              /*const int      s0,*/
                                const int      s1,
                                const int      s2,
                                const int      s3,
                                const int      s00,
                                const int      s01,
                                const int      s02,
                                const int      s03,
                                const int      s10,
                                const int      s11,
                                const int      s12,
                                const int      s13,
                                const sycl::nd_item<1> & item) {

    const uint32_t i = item.get_global_id(0);

    const uint32_t i3 = i / prod_012;
    const uint32_t i2 = (i - i3 * prod_012) / prod_01;
    const uint32_t i1 = (i - i3 * prod_012 - i2 * prod_01) / ne0;
    const uint32_t i0 = i - i3 * prod_012 - i2 * prod_01 - i1 * ne0;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const uint32_t i11 = i1 % ne11;
    const uint32_t i12 = i2 % ne12;
    const uint32_t i13 = i3 % ne13;

    const size_t i_src0 = (size_t)i3*s03 + (size_t)i2*s02 + (size_t)i1*s01;
    const size_t i_src1 = (size_t)i13*s13 + (size_t)i12*s12 + (size_t)i11*s11;
    const size_t i_dst  = (size_t)i3*s3  + (size_t)i2*s2  + (size_t)i1*s1;

    const src0_t * src0_row = src0 ? (src0 + i_src0) : nullptr;
    dst_t * dst_row = dst + i_dst;

    const uint32_t i10 = i0 % ne10;

    float result = src0_row ? (float)src0_row[i0*s00] : 0.0f;
    result = apply_bin_op<op>(result, (float)src1[i_src1 + i10*s10]);

    dst_row[i0] = (dst_t)result;
}

// ============================================================================
// Launch function — host side
// ============================================================================

template <bin_op_type op, typename src0_t, typename src1_t, typename dst_t>
static void launch_bin_bcast(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                             const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
                             sycl::queue & stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    int nr0 = ne10 / ne0;
    int nr1 = ne11 / ne1;
    int nr2 = ne12 / ne2;
    int nr3 = ne13 / ne3;

    int nr[4] = { nr0, nr1, nr2, nr3 };

    int64_t cne[]  = { ne0, ne1, ne2, ne3 };
    int64_t cne0[] = { ne00, ne01, ne02, ne03 };
    int64_t cne1[] = { ne10, ne11, ne12, ne13 };

    size_t cnb[]  = { nb0, nb1, nb2, nb3 };
    size_t cnb0[] = { nb00, nb01, nb02, nb03 };
    size_t cnb1[] = { nb10, nb11, nb12, nb13 };

    auto collapse = [](int64_t cne[]) {
        cne[0] *= cne[1];
        cne[1] = cne[2];
        cne[2] = cne[3];
        cne[3] = 1;
    };

    auto collapse_nb = [](size_t cnb[], const int64_t cne[]) {
        cnb[1] *= cne[1];
        cnb[2] *= cne[2];
        cnb[3] *= cne[3];
    };

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && !ggml_is_permuted(src0) && !ggml_is_permuted(src1)) {
        for (int i = 0; i < 4; i++) {
            if (nr[i] != 1) {
                break;
            }
            if (i > 0) {
                collapse_nb(cnb, cne);
                collapse_nb(cnb0, cne0);
                collapse_nb(cnb1, cne1);
                collapse(cne);
                collapse(cne0);
                collapse(cne1);
            }
        }
    }

    {
        int64_t ne0 = cne[0];
        int64_t ne1 = cne[1];
        int64_t ne2 = cne[2];
        int64_t ne3 = cne[3];

        size_t nb0 = cnb[0];
        size_t nb1 = cnb[1];
        size_t nb2 = cnb[2];
        size_t nb3 = cnb[3];

        size_t nb00 = cnb0[0];
        size_t nb01 = cnb0[1];
        size_t nb02 = cnb0[2];
        size_t nb03 = cnb0[3];

        size_t nb10 = cnb1[0];
        size_t nb11 = cnb1[1];
        size_t nb12 = cnb1[2];
        size_t nb13 = cnb1[3];

      //size_t s0 = nb0 / sizeof(dst_t);
        size_t s1 = nb1 / sizeof(dst_t);
        size_t s2 = nb2 / sizeof(dst_t);
        size_t s3 = nb3 / sizeof(dst_t);

        size_t s10 = nb10 / sizeof(src1_t);
        size_t s11 = nb11 / sizeof(src1_t);
        size_t s12 = nb12 / sizeof(src1_t);
        size_t s13 = nb13 / sizeof(src1_t);

        size_t s00 = nb00 / sizeof(src0_t);
        size_t s01 = nb01 / sizeof(src0_t);
        size_t s02 = nb02 / sizeof(src0_t);
        size_t s03 = nb03 / sizeof(src0_t);

        GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

        GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

        GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

        const int block_size = 256;

        int64_t hne0 = std::max(ne0 / 2LL, 1LL);

        // SYCL nd_range uses {z, y, x} ordering (range<3>(z, y, x))
        unsigned int bx = std::min<unsigned int>(hne0, block_size);
        unsigned int by = std::min<unsigned int>(ne1, block_size / bx);
        unsigned int bz = std::min(std::min<unsigned int>(ne2 * ne3, block_size / bx / by), 64U);

        unsigned int gx = (hne0 + bx - 1) / bx;
        unsigned int gy = (ne1 + by - 1) / by;
        unsigned int gz = (ne2 * ne3 + bz - 1) / bz;

        const uint32_t cne10 = (uint32_t) cne1[0];
        const uint32_t cne11 = (uint32_t) cne1[1];
        const uint32_t cne12 = (uint32_t) cne1[2];
        const uint32_t cne13 = (uint32_t) cne1[3];

        if (gz > 65535 || gy > 65535) {
            // Fallback to 1D kernel
            int total = ne0 * ne1 * ne2 * ne3;
            int block_num = (total + block_size - 1) / block_size;
            const uint32_t prod_012 = (uint32_t)(ne0 * ne1 * ne2);
            const uint32_t prod_01  = (uint32_t)(ne0 * ne1);

            stream.parallel_for(
                sycl::nd_range<1>(block_num * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    k_bin_bcast_unravel<op>(
                        src0_dd, src1_dd, dst_dd,
                        (uint32_t)ne0, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                        prod_012, prod_01,
                        cne10, cne11, cne12, cne13,
                      /*s0,*/ s1, s2, s3,
                        s00, s01, s02, s03,
                        s10, s11, s12, s13,
                        item);
                });
        } else {
            // 3D kernel — SYCL nd_range<3> uses (z, y, x) ordering
            sycl::range<3> block_dims(bz, by, bx);
            sycl::range<3> grid_dims(gz * bz, gy * by, gx * bx);

            stream.parallel_for(
                sycl::nd_range<3>(grid_dims, block_dims),
                [=](sycl::nd_item<3> item) {
                    k_bin_bcast<op>(
                        src0_dd, src1_dd, dst_dd,
                        (int)ne0, (int)ne1, (int)ne2, (uint32_t)ne3,
                        cne10, cne11, cne12, cne13,
                      /*s0,*/ s1, s2, s3,
                        s00, s01, s02, s03,
                        s10, s11, s12, s13,
                        item);
                });
        }
    }
}

// ============================================================================
// Type dispatch
// ============================================================================

template <bin_op_type op>
static void ggml_sycl_op_bin_bcast(
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const void * src0_dd, const void * src1_dd, void * dst_dd, sycl::queue & stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        launch_bin_bcast<op>(src0, src1, dst,
            (const float *)src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        launch_bin_bcast<op>(src0, src1, dst,
            (const sycl::half *)src0_dd, (const sycl::half *)src1_dd, (sycl::half *)dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        launch_bin_bcast<op>(src0, src1, dst,
            (const sycl::half *)src0_dd, (const float *)src1_dd, (sycl::half *)dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        launch_bin_bcast<op>(src0, src1, dst,
            (const sycl::half *)src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

// ============================================================================
// Public API — concrete wrappers
// ============================================================================

void ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    // For repeat: src0 is the pattern (smaller), dst is the output (larger).
    // CUDA passes: (dst, dst->src[0], dst, nullptr, src[0]->data, dst->data)
    // i.e. src0_tensor=dst (for shape), src1_tensor=src[0] (data to repeat),
    //      src0_dd=nullptr, src1_dd=src[0]->data, dst_dd=dst->data
    sycl::queue & stream = *(ctx.stream());
    ggml_sycl_op_bin_bcast<bin_op_type::REPEAT>(
        dst, dst->src[0], dst,
        nullptr, dst->src[0]->data, dst->data, stream);
}

void ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue & stream = *(ctx.stream());
    ggml_sycl_op_bin_bcast<bin_op_type::ADD>(
        dst->src[0], dst->src[1], dst,
        dst->src[0]->data, dst->src[1]->data, dst->data, stream);
}

void ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue & stream = *(ctx.stream());
    ggml_sycl_op_bin_bcast<bin_op_type::SUB>(
        dst->src[0], dst->src[1], dst,
        dst->src[0]->data, dst->src[1]->data, dst->data, stream);
}

void ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue & stream = *(ctx.stream());
    ggml_sycl_op_bin_bcast<bin_op_type::MUL>(
        dst->src[0], dst->src[1], dst,
        dst->src[0]->data, dst->src[1]->data, dst->data, stream);
}

void ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue & stream = *(ctx.stream());
    ggml_sycl_op_bin_bcast<bin_op_type::DIV>(
        dst->src[0], dst->src[1], dst,
        dst->src[0]->data, dst->src[1]->data, dst->data, stream);
}
