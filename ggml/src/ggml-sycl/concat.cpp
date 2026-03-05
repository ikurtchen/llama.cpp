#include "concat.hpp"

#include <cstdint>

// ============================================================================
// Contiguous kernels
// ============================================================================

static void concat_f32_sycl_dim0(
        const float * x, const float * y, float * dst,
        const int ne0, const int ne00,
        const sycl::nd_item<3> & item) {
    int nidx = item.get_global_id(2);
    if (nidx >= ne0) {
        return;
    }

    // item.get_group(1) = row index (ne1 dimension)
    // item.get_group(0) = slice index (ne2 dimension)
    int row   = item.get_group(1);
    int slice = item.get_group(0);
    int grid_dim_y = item.get_group_range(1);

    int offset_dst =
        nidx +
        row * ne0 +
        slice * ne0 * grid_dim_y;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            row * ne00 +
            slice * ne00 * grid_dim_y;
        dst[offset_dst] = x[offset_src];
    } else { // src1
        int offset_src =
            (nidx - ne00) +
            row * (ne0 - ne00) +
            slice * (ne0 - ne00) * grid_dim_y;
        dst[offset_dst] = y[offset_src];
    }
}

static void concat_f32_sycl_dim1(
        const float * x, const float * y, float * dst,
        const int ne0, const int ne01,
        const sycl::nd_item<3> & item) {
    int nidx = item.get_global_id(2);
    if (nidx >= ne0) {
        return;
    }

    int row   = item.get_group(1);
    int slice = item.get_group(0);
    int grid_dim_y = item.get_group_range(1);

    int offset_dst =
        nidx +
        row * ne0 +
        slice * ne0 * grid_dim_y;

    if (row < ne01) { // src0
        int offset_src =
            nidx +
            row * ne0 +
            slice * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else { // src1
        int offset_src =
            nidx +
            (row - ne01) * ne0 +
            slice * ne0 * (grid_dim_y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

static void concat_f32_sycl_dim2(
        const float * x, const float * y, float * dst,
        const int ne0, const int ne02,
        const sycl::nd_item<3> & item) {
    int nidx = item.get_global_id(2);
    if (nidx >= ne0) {
        return;
    }

    int row   = item.get_group(1);
    int slice = item.get_group(0);
    int grid_dim_y = item.get_group_range(1);

    int offset_dst =
        nidx +
        row * ne0 +
        slice * ne0 * grid_dim_y;

    if (slice < ne02) { // src0
        int offset_src =
            nidx +
            row * ne0 +
            slice * ne0 * grid_dim_y;
        dst[offset_dst] = x[offset_src];
    } else { // src1
        int offset_src =
            nidx +
            row * ne0 +
            (slice - ne02) * ne0 * grid_dim_y;
        dst[offset_dst] = y[offset_src];
    }
}

// Host-side dispatch for contiguous dim 0/1/2 kernels
static void concat_f32_sycl(
        const float * x, const float * y, float * dst,
        int ne00, int ne01, int ne02,
        int ne0, int ne1, int ne2, int dim,
        sycl::queue & stream) {
    int num_blocks = (ne0 + SYCL_CONCAT_BLOCK_SIZE - 1) / SYCL_CONCAT_BLOCK_SIZE;
    // nd_range<3> uses (dim0, dim1, dim2) ordering.
    // Map: dim0 = ne2 (slice/blockIdx.z), dim1 = ne1 (row/blockIdx.y), dim2 = columns (blockIdx.x * blockDim.x)
    sycl::range<3> global(ne2, ne1, num_blocks * SYCL_CONCAT_BLOCK_SIZE);
    sycl::range<3> local(1, 1, SYCL_CONCAT_BLOCK_SIZE);

    if (dim == 0) {
        stream.parallel_for(
            sycl::nd_range<3>(global, local),
            [=](sycl::nd_item<3> item) {
                concat_f32_sycl_dim0(x, y, dst, ne0, ne00, item);
            });
        return;
    }
    if (dim == 1) {
        stream.parallel_for(
            sycl::nd_range<3>(global, local),
            [=](sycl::nd_item<3> item) {
                concat_f32_sycl_dim1(x, y, dst, ne0, ne01, item);
            });
        return;
    }
    // dim == 2
    stream.parallel_for(
        sycl::nd_range<3>(global, local),
        [=](sycl::nd_item<3> item) {
            concat_f32_sycl_dim2(x, y, dst, ne0, ne02, item);
        });
}

// ============================================================================
// Non-contiguous kernel (slow)
// ============================================================================

template <int dim>
static void concat_f32_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
        int64_t   ne00,
        int64_t   ne01,
        int64_t   ne02,
        int64_t   ne03,
        uint64_t  nb00,
        uint64_t  nb01,
        uint64_t  nb02,
        uint64_t  nb03,
        int64_t   /*ne10*/,
        int64_t   /*ne11*/,
        int64_t   /*ne12*/,
        int64_t   /*ne13*/,
        uint64_t  nb10,
        uint64_t  nb11,
        uint64_t  nb12,
        uint64_t  nb13,
        int64_t   ne0,
        int64_t   /*ne1*/,
        int64_t   /*ne2*/,
        int64_t   /*ne3*/,
        uint64_t  nb0,
        uint64_t  nb1,
        uint64_t  nb2,
        uint64_t  nb3,
        const sycl::nd_item<3> & item) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    // Map: group(0) = i3, group(1) = i2, local_id(2) loops over i0
    // Grid was launched as (ne3, ne2, ne1 * BLOCK_SIZE) with local (1, 1, BLOCK_SIZE)
    const int64_t i3 = item.get_group(0);
    const int64_t i2 = item.get_group(1);
    const int64_t i1 = item.get_group(2);

    const float * x;

    for (int64_t i0 = item.get_local_id(2); i0 < ne0; i0 += item.get_local_range(2)) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const float *)(src0 + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const float *)(src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
            } else if constexpr (dim == 1) {
                x = (const float *)(src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
            } else if constexpr (dim == 2) {
                x = (const float *)(src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
            } else if constexpr (dim == 3) {
                x = (const float *)(src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
            }
        }

        float * y = (float *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);
        *y = *x;
    }
}

// ============================================================================
// Host dispatch function
// ============================================================================

void ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    sycl::queue & stream = *ctx.stream();

    const int32_t dim = ((const int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const float * src0_d = (const float *) src0->data;
        const float * src1_d = (const float *) src1->data;

        float * dst_d = (float *) dst->data;

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_f32_sycl(
                    src0_d + i3 * (src0->nb[3] / 4),
                    src1_d + i3 * (src1->nb[3] / 4),
                    dst_d  + i3 * ( dst->nb[3] / 4),
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    dst->ne[0],  dst->ne[1],  dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            stream.memcpy(dst_d,            src0_d, size0).wait();
            stream.memcpy(dst_d + size0 / 4, src1_d, size1).wait();
        }
    } else {
        // Non-contiguous path
        // CUDA grid: dim3(ne1, ne2, ne3) with blockDim = BLOCK_SIZE
        // SYCL: nd_range<3> global = (ne3, ne2, ne1 * BLOCK_SIZE), local = (1, 1, BLOCK_SIZE)

        // Extract all tensor fields into local variables before the lambda.
        // The ggml_tensor structs live in host memory and CANNOT be dereferenced
        // inside a SYCL device kernel — we must capture plain values.
        const char * src0_d = (const char *) src0->data;
        const char * src1_d = (const char *) src1->data;
        char       * dst_d  = (      char *) dst->data;

        const int64_t s0_ne0 = src0->ne[0]; const int64_t s0_ne1 = src0->ne[1];
        const int64_t s0_ne2 = src0->ne[2]; const int64_t s0_ne3 = src0->ne[3];
        const uint64_t s0_nb0 = src0->nb[0]; const uint64_t s0_nb1 = src0->nb[1];
        const uint64_t s0_nb2 = src0->nb[2]; const uint64_t s0_nb3 = src0->nb[3];

        const int64_t s1_ne0 = src1->ne[0]; const int64_t s1_ne1 = src1->ne[1];
        const int64_t s1_ne2 = src1->ne[2]; const int64_t s1_ne3 = src1->ne[3];
        const uint64_t s1_nb0 = src1->nb[0]; const uint64_t s1_nb1 = src1->nb[1];
        const uint64_t s1_nb2 = src1->nb[2]; const uint64_t s1_nb3 = src1->nb[3];

        const int64_t d_ne0 = dst->ne[0]; const int64_t d_ne1 = dst->ne[1];
        const int64_t d_ne2 = dst->ne[2]; const int64_t d_ne3 = dst->ne[3];
        const uint64_t d_nb0 = dst->nb[0]; const uint64_t d_nb1 = dst->nb[1];
        const uint64_t d_nb2 = dst->nb[2]; const uint64_t d_nb3 = dst->nb[3];

        sycl::range<3> global(d_ne3, d_ne2, d_ne1 * SYCL_CONCAT_BLOCK_SIZE);
        sycl::range<3> local(1, 1, SYCL_CONCAT_BLOCK_SIZE);

        auto launch_kernel = [&](auto dim_constant) {
            stream.parallel_for(
                sycl::nd_range<3>(global, local),
                [=](sycl::nd_item<3> item) {
                    concat_f32_non_cont<dim_constant>(
                        src0_d, src1_d, dst_d,
                        s0_ne0, s0_ne1, s0_ne2, s0_ne3,
                        s0_nb0, s0_nb1, s0_nb2, s0_nb3,
                        s1_ne0, s1_ne1, s1_ne2, s1_ne3,
                        s1_nb0, s1_nb1, s1_nb2, s1_nb3,
                        d_ne0,  d_ne1,  d_ne2,  d_ne3,
                        d_nb0,  d_nb1,  d_nb2,  d_nb3,
                        item);
                });
        };

        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}
