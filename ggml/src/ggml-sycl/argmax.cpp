// SYCL argmax op: per-row index of the maximum value. One work-group per row; each work-item
// scans a strided slice of the row, packs (value, index) into a single sortable 64-bit key
// (value's monotonic bit-ordering in the high 32 bits, index in the low 32 bits), then a
// work-group reduction (sycl::reduce_over_group with sycl::maximum - the idiomatic group
// collective, not a serial loop) finds the row's max key in one step. Mirrors
// ggml-cuda/argmax.cu's algorithm but written directly against SYCL.
#include "argmax.hpp"

#include <sycl/sycl.hpp>
#include <cfloat>
#include <cstdint>

namespace {
// Maps a float to a uint32_t that preserves ordering (matches ggml-cuda/top-k.cu's
// top_k_float_to_ordered): flips the sign bit for positives, inverts all bits for negatives.
inline uint32_t float_to_ordered(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}
}  // namespace

static void argmax_f32(sycl::queue & q, const float * x, int32_t * dst, int64_t ncols, int64_t nrows,
                       size_t wg_size) {
    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(nrows * wg_size), sycl::range<1>(wg_size)),
            [=](sycl::nd_item<1> item) {
            const int64_t row = item.get_group(0);
            const float * rowx = x + row * ncols;

            uint64_t best_key = 0;  // ordered_value(0) << 32 | 0, safe since it is overwritten below
            bool     has_val  = false;
            for (int64_t col = item.get_local_id(0); col < ncols; col += wg_size) {
                const uint64_t key = ((uint64_t) float_to_ordered(rowx[col]) << 32) | (uint32_t) col;
                if (!has_val || key > best_key) {
                    best_key = key;
                    has_val  = true;
                }
            }

            const uint64_t result = sycl::reduce_over_group(item.get_group(), best_key, sycl::maximum<uint64_t>());
            if (item.get_local_id(0) == 0) {
                dst[row] = (int32_t) (uint32_t) (result & 0xFFFFFFFFu);
            }
        });
    });
}

void ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const size_t max_wg = ctx.stream().get_device().get_info<sycl::info::device::max_work_group_size>();
    size_t wg_size = std::min<size_t>(max_wg, ((size_t) ne00 + 31) / 32 * 32);
    wg_size = std::max<size_t>(wg_size, 32);

    sycl::queue & q = ctx.stream();
    argmax_f32(q, (const float *) src0->data, (int32_t *) dst->data, ne00, nrows, wg_size);
}

bool ggml_sycl_supports_argmax(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 && ggml_is_contiguous(src0);
}
