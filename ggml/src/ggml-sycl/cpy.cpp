#include "cpy.hpp"

#include <sycl/sycl.hpp>
#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #include <sycl/ext/oneapi/bfloat16.hpp>
#endif
#include <cstring>

#include "dequantize.hpp"
#include "ggml-sycl/presets.hpp"

using sycl_bf16 = sycl::ext::oneapi::bfloat16;

// ============================================================
// Block size for contiguous copy kernels (larger = better occupancy for simple copies)
// The generic non-contiguous kernel uses SYCL_CPY_BLOCK_SIZE (32) from presets.hpp
// ============================================================
constexpr int CPY_CONTIGUOUS_BLOCK_SIZE = 256;

// ============================================================
// Constants for SLM-tiled 2D transpose kernel
// ============================================================
constexpr int CPY_TILE_DIM   = 32;   // tile width/height in elements
constexpr int CPY_BLOCK_ROWS = 8;    // rows per thread (TILE_DIM / BLOCK_ROWS threads in y)
constexpr int CPY_BLOCK_NM   = 8;    // matrices batched per workgroup z-slice

// ============================================================
// Kernel: cpy_scalar_transpose — SLM-tiled 2D transpose
// ============================================================
// Transposes a 2D matrix (ne01 columns × ne00 rows in src → ne00 columns × ne01 rows in dst)
// using shared local memory to coalesce both reads and writes.
// SLM tile is float[TILE_DIM][TILE_DIM+1]; for sub-32-bit types (F16/BF16), values are
// packed via reinterpret_cast (2 half values per float slot), matching CUDA cpy_scalar_transpose.
template <typename T>
static void k_cpy_scalar_transpose(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01,
    const sycl::nd_item<3> & item,
    float * tile_data)  // pointer to SLM: TILE_DIM * (TILE_DIM+1) floats
{
    const T * src = reinterpret_cast<const T *>(cx);
    T * dst       = reinterpret_cast<T *>(cdst);

    const int64_t nmat = ne / (ne00 * ne01);
    const int64_t n    = ne00 * ne01;

    // SYCL nd_range<3>: group(0)=z, group(1)=y, group(2)=x
    const int x  = (int)item.get_group(2) * CPY_TILE_DIM + (int)item.get_local_id(2);
    const int y  = (int)item.get_group(1) * CPY_TILE_DIM + (int)item.get_local_id(1);
    const int tx = (int)item.get_group(1) * CPY_TILE_DIM + (int)item.get_local_id(2);  // transposed block offset
    const int ty = (int)item.get_group(2) * CPY_TILE_DIM + (int)item.get_local_id(1);

    // Access SLM tile as tile[row][col] with col stride = TILE_DIM+1 to avoid bank conflicts
    constexpr int TILE_STRIDE = CPY_TILE_DIM + 1;

    for (int i = 0; i < CPY_BLOCK_NM; ++i) {
        const int imat = (int)item.get_group(0) * CPY_BLOCK_NM + i;
        if (imat >= nmat) break;

        // Load tile from src (coalesced read along columns of the source matrix)
        for (int j = 0; j < CPY_TILE_DIM; j += CPY_BLOCK_ROWS) {
            if (x < ne01 && (y + j) < ne00) {
                const int row = (int)item.get_local_id(1) + j;
                const int col = (int)item.get_local_id(2) * (int)(sizeof(float) / sizeof(T));
                T * tile_row = reinterpret_cast<T *>(&tile_data[row * TILE_STRIDE]);
                tile_row[col] = src[imat * n + (int64_t)(y + j) * ne01 + x];
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        // Store tile to dst (coalesced write, transposed)
        for (int j = 0; j < CPY_TILE_DIM; j += CPY_BLOCK_ROWS) {
            if ((ty + j) < ne01 && tx < ne00) {
                const int col = ((int)item.get_local_id(1) + j) * (int)(sizeof(float) / sizeof(T));
                const T * tile_row = reinterpret_cast<const T *>(&tile_data[(int)item.get_local_id(2) * TILE_STRIDE]);
                dst[imat * n + (int64_t)(ty + j) * ne00 + tx] = tile_row[col];
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
}

// ============================================================
// Launch helper: cpy_scalar_transpose
// ============================================================
template <typename T>
static void launch_cpy_scalar_transpose(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb02,
    sycl::queue & stream)
{
    // Flatten batch dimensions if strides allow: when nb00 <= nb02, batches
    // along dim2 are contiguous and can be merged with dim1.
    int64_t ne00n, ne01n;
    if (nb00 <= nb02) {
        ne00n = ne00;
        ne01n = ne01;
    } else {
        ne00n = ne00;
        ne01n = ne01 * ne02;
    }
    const int64_t nmat = ne / (ne00n * ne01n);

    const int64_t grid_x = (ne01n + CPY_TILE_DIM - 1) / CPY_TILE_DIM;
    const int64_t grid_y = (ne00n + CPY_TILE_DIM - 1) / CPY_TILE_DIM;
    const int64_t grid_z = (nmat  + CPY_BLOCK_NM  - 1) / CPY_BLOCK_NM;

    const sycl::range<3> global(grid_z, grid_y * CPY_BLOCK_ROWS, grid_x * CPY_TILE_DIM);
    const sycl::range<3> local(1, CPY_BLOCK_ROWS, CPY_TILE_DIM);

    // Capture ne00n and ne01n for the kernel (these may differ from the original ne00/ne01)
    const int64_t k_ne00 = ne00n;
    const int64_t k_ne01 = ne01n;

    stream.submit([&](sycl::handler & cgh) {
        // Allocate SLM tile: TILE_DIM rows × (TILE_DIM+1) columns of float
        sycl::local_accessor<float, 1> tile_acc(sycl::range<1>(CPY_TILE_DIM * (CPY_TILE_DIM + 1)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(global, local),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                float * tile_ptr = tile_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                k_cpy_scalar_transpose<T>(cx, cdst, ne, k_ne00, k_ne01, item, tile_ptr);
            });
    });
}

// ============================================================
// Per-element copy callbacks (from baseline) — used as function-pointer
// template arguments to cpy_f32_f16<cpy_1>.  Keeping the baseline's
// function-pointer pattern produces better codegen on Intel GPUs than
// a direct template<src_t,dst_t> approach.
// ============================================================
static void cpy_1_f32_f32(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    float *       dsti = (float *) cdsti;
    *dsti = *xi;
}

static void cpy_1_f32_f16(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    sycl::half *  dsti = (sycl::half *) cdsti;
    *dsti = sycl::vec<float, 1>(*xi).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

static void cpy_1_f16_f16(const char * cxi, char * cdsti) {
    const sycl::half * xi   = (const sycl::half *) cxi;
    sycl::half *       dsti = (sycl::half *) cdsti;
    *dsti = *xi;
}

static void cpy_1_f16_f32(const char * cxi, char * cdsti) {
    const sycl::half * xi   = (const sycl::half *) cxi;
    float *            dsti = (float *) cdsti;
    *dsti = *xi;
}

static void cpy_1_i32_i32(const char * cxi, char * cdsti) {
    const int32_t * xi   = (const int32_t *) cxi;
    int32_t *       dsti = (int32_t *) cdsti;
    *dsti = *xi;
}

static void cpy_1_f32_i32(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    int32_t *     dsti = (int32_t *) cdsti;
    *dsti = (int32_t)(*xi);
}

static void cpy_1_i32_f32(const char * cxi, char * cdsti) {
    const int32_t * xi   = (const int32_t *) cxi;
    float *         dsti = (float *) cdsti;
    *dsti = (float)(*xi);
}

static void cpy_1_bf16_bf16(const char * cxi, char * cdsti) {
    const sycl_bf16 * xi   = (const sycl_bf16 *) cxi;
    sycl_bf16 *       dsti = (sycl_bf16 *) cdsti;
    *dsti = *xi;
}

static void cpy_1_bf16_f32(const char * cxi, char * cdsti) {
    const sycl_bf16 * xi   = (const sycl_bf16 *) cxi;
    float *           dsti = (float *) cdsti;
    *dsti = (float)(*xi);
}

static void cpy_1_f32_bf16(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    sycl_bf16 *   dsti = (sycl_bf16 *) cdsti;
    *dsti = sycl_bf16(*xi);
}

static void cpy_1_bf16_f16(const char * cxi, char * cdsti) {
    const sycl_bf16 * xi   = (const sycl_bf16 *) cxi;
    sycl::half *      dsti = (sycl::half *) cdsti;
    *dsti = sycl::half((float)(*xi));
}

static void cpy_1_f16_bf16(const char * cxi, char * cdsti) {
    const sycl::half * xi   = (const sycl::half *) cxi;
    sycl_bf16 *        dsti = (sycl_bf16 *) cdsti;
    *dsti = sycl_bf16((float)(*xi));
}

// ============================================================
// Kernel: k_cpy_scalar — generic non-contiguous element copy
// ============================================================
// Uses the baseline's function-pointer template pattern for best codegen.
// Maps 1D flat index i → 4D coordinates → byte offsets for both src and dst.
// Uses int (32-bit) arithmetic — 64-bit integer division is much slower on GPU.
template <cpy_kernel_t cpy_1>
static void k_cpy_scalar(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    const sycl::nd_item<3> & item_ct1) {

    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
    if (i >= ne) return;

    // 4D index decomposition for source
    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    // 4D index decomposition for destination
    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
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
// Quantized copy kernels (from baseline)
// ============================================================

// f32 → quantized block copy (one block per work-item)
template <cpy_kernel_t cpy_blck, int qk>
static void k_cpy_f32_q(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    const sycl::nd_item<3> & item_ct1) {

    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;
    if (i >= ne) return;

    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

// quantized block → f32 copy (one block per work-item)
template <cpy_kernel_t cpy_blck, int qk>
static void k_cpy_q_f32(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    const sycl::nd_item<3> & item_ct1) {

    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;
    if (i >= ne) return;

    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

// Same quantized type block copy
template<typename T>
static void cpy_blck_q_q(const char * cxi, char * cdsti) {
    const T * xi = (const T *) cxi;
    T * dsti = (T *) cdsti;
    *dsti = *xi;
}

template <typename T, int qk>
static void k_cpy_q_q(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    const sycl::nd_item<3> & item_ct1) {

    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;
    if (i >= ne) return;

    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck_q_q<T>(cx + x_offset, cdst + dst_offset);
}

// q8_0 → f32 dequantize block copy
static void cpy_blck_q8_0_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *) (cdsti);
    for (int j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j)     = dq.x();
        *(cdstf + j + 1) = dq.y();
    }
}

// Generic quantized → f32 dequantize block copy
template <dequantize_kernel_t dequant, int qk>
static void cpy_blck_q_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *) (cdsti);
    for (int j = 0; j < qk / 2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j)          = dq.x();
        *(cdstf + j + qk / 2) = dq.y();
    }
}

// ============================================================
// Launch helpers
// ============================================================

// Contiguous copy uses larger block size (256) for better throughput
template<typename src_t, typename dst_t>
static void launch_cpy_scalar_contiguous(
    const char * cx, char * cdst, const int64_t ne, sycl::queue & stream) {

    const int64_t num_blocks = (ne + CPY_CONTIGUOUS_BLOCK_SIZE - 1) / CPY_CONTIGUOUS_BLOCK_SIZE;
    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * CPY_CONTIGUOUS_BLOCK_SIZE, CPY_CONTIGUOUS_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            k_cpy_scalar_contiguous<src_t, dst_t>(cx, cdst, ne, item);
        });
}

// Generic non-contiguous copy uses SYCL_CPY_BLOCK_SIZE (32) and nd_range<3>
// matching baseline kernel signature (function-pointer template, int parameters,
// 3D dispatch) for best codegen on Intel GPUs.
template <cpy_kernel_t cpy_1>
static void launch_cpy_scalar(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    sycl::queue & stream) {

    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            k_cpy_scalar<cpy_1>(cx, cdst, ne,
                ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

// ============================================================
// Quantized launch helpers (from baseline, using nd_range<3>)
// ============================================================

static void launch_cpy_f32_q8_0(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    sycl::queue * stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    const int num_blocks = ne / QK8_0;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            k_cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                   ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void launch_cpy_q8_0_f32(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    sycl::queue * stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            k_cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                   ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void launch_cpy_f32_q4_0(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    sycl::queue * stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ne / QK4_0;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            k_cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                   ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void launch_cpy_q4_0_f32(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    sycl::queue * stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            k_cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                        nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                        item_ct1);
        });
}

static void launch_cpy_f32_q4_1(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    sycl::queue * stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ne / QK4_1;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            k_cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                   ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void launch_cpy_q4_1_f32(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    sycl::queue * stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            k_cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                        nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                        item_ct1);
        });
}

static void launch_cpy_f32_q5_0(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    sycl::queue * stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    const int num_blocks = ne / QK5_0;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            k_cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                   ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void launch_cpy_q5_0_f32(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    sycl::queue * stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            k_cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                        nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                        item_ct1);
        });
}

static void launch_cpy_f32_q5_1(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    sycl::queue * stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    const int num_blocks = ne / QK5_1;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            k_cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                   ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void launch_cpy_q5_1_f32(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    sycl::queue * stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            k_cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                        nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                        item_ct1);
        });
}

static void launch_cpy_f32_iq4_nl(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    sycl::queue * stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    const int num_blocks = ne / QK4_NL;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            k_cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                      ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

// Same quantized type copy launch helpers
template <typename T, int qk>
static void launch_cpy_q_q(const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    sycl::queue * stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            k_cpy_q_q<T, qk>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                              ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
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

    // Detect simple 2D transpose: dim1 stride == element_size (contiguous along dim1),
    // ne[3]==1, and dim2 stride == ne00*ne01*element_size (batches are contiguous).
    // Destination must be fully contiguous. Matches CUDA's can_be_transposed check.
    // Additional heuristic: only use SLM-tiled transpose when the aspect ratio
    // ne00:ne01 is not too extreme — very wide, short matrices (e.g. 786432×256)
    // are faster with the generic scalar kernel.
    const bool can_be_transposed = !contiguous_srcs &&
        src0->ne[3] == 1 &&
        nb01 == (int64_t)ggml_element_size(src0) &&
        nb02 == ne00 * ne01 * (int64_t)ggml_element_size(src0) &&
        ggml_is_contiguous(src1) &&
        ne00 <= ne01 * 128;

    // f32 → f32
    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<float, float>(src0_ddc, src1_ddc, ne, stream);
        } else if (can_be_transposed) {
            launch_cpy_scalar_transpose<float>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb02, stream);
        } else {
            launch_cpy_scalar<cpy_1_f32_f32>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // f32 → f16
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<float, sycl::half>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<cpy_1_f32_f16>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // f16 → f32
    else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<sycl::half, float>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<cpy_1_f16_f32>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // f16 → f16
    else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<sycl::half, sycl::half>(src0_ddc, src1_ddc, ne, stream);
        } else if (can_be_transposed) {
            launch_cpy_scalar_transpose<sycl::half>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb02, stream);
        } else {
            launch_cpy_scalar<cpy_1_f16_f16>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // f32 → i32
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<float, int32_t>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<cpy_1_f32_i32>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // i32 → f32
    else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<int32_t, float>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<cpy_1_i32_f32>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // i32 → i32 (non-contiguous; contiguous already handled by memcpy above)
    else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        if (can_be_transposed) {
            launch_cpy_scalar_transpose<int32_t>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb02, stream);
        } else {
            launch_cpy_scalar<cpy_1_i32_i32>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // bf16 → bf16
    else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<sycl_bf16, sycl_bf16>(src0_ddc, src1_ddc, ne, stream);
        } else if (can_be_transposed) {
            launch_cpy_scalar_transpose<sycl_bf16>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb02, stream);
        } else {
            launch_cpy_scalar<cpy_1_bf16_bf16>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // bf16 → f32
    else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<sycl_bf16, float>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<cpy_1_bf16_f32>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // f32 → bf16
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<float, sycl_bf16>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<cpy_1_f32_bf16>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // bf16 → f16
    else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<sycl_bf16, sycl::half>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<cpy_1_bf16_f16>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // f16 → bf16
    else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            launch_cpy_scalar_contiguous<sycl::half, sycl_bf16>(src0_ddc, src1_ddc, ne, stream);
        } else {
            launch_cpy_scalar<cpy_1_f16_bf16>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
        }
    }
    // f32 → q8_0
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        launch_cpy_f32_q8_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // q8_0 → f32
    else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        launch_cpy_q8_0_f32(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // f32 → q4_0
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        launch_cpy_f32_q4_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // q4_0 → f32
    else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        launch_cpy_q4_0_f32(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // f32 → q4_1
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        launch_cpy_f32_q4_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // q4_1 → f32
    else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        launch_cpy_q4_1_f32(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // f32 → q5_0
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        launch_cpy_f32_q5_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // q5_0 → f32
    else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        launch_cpy_q5_0_f32(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // f32 → q5_1
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        launch_cpy_f32_q5_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // q5_1 → f32
    else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        launch_cpy_q5_1_f32(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // f32 → iq4_nl
    else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        launch_cpy_f32_iq4_nl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // q8_0 → q8_0
    else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        launch_cpy_q_q<block_q8_0, QK8_0>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // q4_0 → q4_0
    else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_Q4_0) {
        launch_cpy_q_q<block_q4_0, QK4_0>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // q4_1 → q4_1
    else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_Q4_1) {
        launch_cpy_q_q<block_q4_1, QK4_1>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // q5_0 → q5_0
    else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_Q5_0) {
        launch_cpy_q_q<block_q5_0, QK5_0>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
    }
    // q5_1 → q5_1
    else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_Q5_1) {
        launch_cpy_q_q<block_q5_1, QK5_1>(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, &stream);
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
