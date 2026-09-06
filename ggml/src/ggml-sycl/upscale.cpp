#include "upscale.hpp"

#include <sycl/sycl.hpp>

namespace {
static inline int max_i32(int a, int b) { return a > b ? a : b; }
static inline int min_i32(int a, int b) { return a < b ? a : b; }
static inline int64_t max_i64(int64_t a, int64_t b) { return a > b ? a : b; }
static inline int64_t min_i64(int64_t a, int64_t b) { return a < b ? a : b; }

static inline float load_f32(const char * base, int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03, int x, int y, int z, int w) {
    return *(const float *) (base + (int64_t) x * nb00 + (int64_t) y * nb01 + (int64_t) z * nb02 + (int64_t) w * nb03);
}

static inline float load_clamped_f32(const char * base, int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03, int x0, int y0, int x_off, int y_off, int ne00_src, int ne01_src, int z, int w) {
    const int x = max_i32(0, min_i32(x0 + x_off, ne00_src - 1));
    const int y = max_i32(0, min_i32(y0 + y_off, ne01_src - 1));
    return load_f32(base, nb00, nb01, nb02, nb03, x, y, z, w);
}

static inline float bicubic_weight1(float x) {
    constexpr float a = -0.75f;
    return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
}

static inline float bicubic_weight2(float x) {
    constexpr float a = -0.75f;
    return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
}

static inline float bicubic_interp(float p0, float p1, float p2, float p3, float x) {
    const float w0 = bicubic_weight2(x + 1.0f);
    const float w1 = bicubic_weight1(x + 0.0f);
    const float w2 = bicubic_weight1(1.0f - x);
    const float w3 = bicubic_weight2(2.0f - x);
    return p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

enum ggml_sycl_upscale_mode {
    GGML_SYCL_UPSCALE_NEAREST,
    GGML_SYCL_UPSCALE_BILINEAR,
    GGML_SYCL_UPSCALE_BILINEAR_ANTIALIAS,
    GGML_SYCL_UPSCALE_BICUBIC,
};

void upscale_dispatch(
        sycl::queue & q,
        const float * src,
        float * dst,
        int64_t nb00,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int ne00_src,
        int ne01_src,
        int ne10_dst,
        int ne11_dst,
        int ne12_dst,
        int ne13_dst,
        float sf0,
        float sf1,
        float sf2,
        float sf3,
        float pixel_offset,
        ggml_sycl_upscale_mode mode) {
    const int64_t total = (int64_t) ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    const int64_t block = 256;
    const int64_t nblk = (total + block - 1) / block;
    const char * base = (const char *) src;

    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
            [=](sycl::nd_item<1> item) {
                const int64_t index = item.get_global_id(0);
                if (index >= total) {
                    return;
                }

                const int i10_dst = index % ne10_dst;
                const int i11_dst = (index / ne10_dst) % ne11_dst;
                const int i12_dst = (index / ((int64_t) ne10_dst * ne11_dst)) % ne12_dst;
                const int i13_dst = index / ((int64_t) ne10_dst * ne11_dst * ne12_dst);

                const int i02_src = (int) (i12_dst / sf2);
                const int i03_src = (int) (i13_dst / sf3);

                float result = 0.0f;
                if (mode == GGML_SYCL_UPSCALE_NEAREST) {
                    const int i00_src = (int) (i10_dst / sf0);
                    const int i01_src = (int) (i11_dst / sf1);
                    result = load_f32(base, nb00, nb01, nb02, nb03, i00_src, i01_src, i02_src, i03_src);
                } else if (mode == GGML_SYCL_UPSCALE_BILINEAR) {
                    const float y_src_f = ((float) i11_dst + pixel_offset) / sf1 - pixel_offset;
                    int y0_src = (int) sycl::floor(y_src_f);
                    int y1_src = y0_src + 1;
                    y0_src = max_i32(0, min_i32(y0_src, ne01_src - 1));
                    y1_src = max_i32(0, min_i32(y1_src, ne01_src - 1));
                    float dy = y_src_f - (float) y0_src;
                    dy = sycl::clamp(dy, 0.0f, 1.0f);

                    const float x_src_f = ((float) i10_dst + pixel_offset) / sf0 - pixel_offset;
                    int x0_src = (int) sycl::floor(x_src_f);
                    int x1_src = x0_src + 1;
                    x0_src = max_i32(0, min_i32(x0_src, ne00_src - 1));
                    x1_src = max_i32(0, min_i32(x1_src, ne00_src - 1));
                    float dx = x_src_f - (float) x0_src;
                    dx = sycl::clamp(dx, 0.0f, 1.0f);

                    const float val_a = load_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src, i02_src, i03_src);
                    const float val_b = load_f32(base, nb00, nb01, nb02, nb03, x1_src, y0_src, i02_src, i03_src);
                    const float val_c = load_f32(base, nb00, nb01, nb02, nb03, x0_src, y1_src, i02_src, i03_src);
                    const float val_d = load_f32(base, nb00, nb01, nb02, nb03, x1_src, y1_src, i02_src, i03_src);

                    result = val_a * (1.0f - dx) * (1.0f - dy) +
                             val_b * dx * (1.0f - dy) +
                             val_c * (1.0f - dx) * dy +
                             val_d * dx * dy;
                } else if (mode == GGML_SYCL_UPSCALE_BILINEAR_ANTIALIAS) {
                    const float y = ((float) i11_dst + pixel_offset) / sf1;
                    const float x = ((float) i10_dst + pixel_offset) / sf0;
                    const float support1 = sycl::fmax(1.0f / sf1, 1.0f);
                    const float invscale1 = 1.0f / support1;
                    const float support0 = sycl::fmax(1.0f / sf0, 1.0f);
                    const float invscale0 = 1.0f / support0;

                    const int64_t x_min = max_i64(0, (int64_t) (x - support0 + pixel_offset));
                    const int64_t x_max = min_i64((int64_t) ne00_src, (int64_t) (x + support0 + pixel_offset));
                    const int64_t y_min = max_i64(0, (int64_t) (y - support1 + pixel_offset));
                    const int64_t y_max = min_i64((int64_t) ne01_src, (int64_t) (y + support1 + pixel_offset));

                    float val = 0.0f;
                    float total_weight = 0.0f;
                    for (int64_t sy = y_min; sy < y_max; ++sy) {
                        const float weight_y = sycl::fmax(1.0f - sycl::fabs((sy - y + pixel_offset) * invscale1), 0.0f);
                        for (int64_t sx = x_min; sx < x_max; ++sx) {
                            const float weight_x = sycl::fmax(1.0f - sycl::fabs((sx - x + pixel_offset) * invscale0), 0.0f);
                            const float weight = weight_x * weight_y;
                            if (weight <= 0.0f) {
                                continue;
                            }
                            val += load_f32(base, nb00, nb01, nb02, nb03, (int) sx, (int) sy, i02_src, i03_src) * weight;
                            total_weight += weight;
                        }
                    }

                    result = total_weight > 0.0f ? (val / total_weight) : 0.0f;
                } else {
                    const float y_src_f = ((float) i11_dst + pixel_offset) / sf1 - pixel_offset;
                    const int y0_src = (int) sycl::floor(y_src_f);
                    const float dy = y_src_f - (float) y0_src;
                    const float x_src_f = ((float) i10_dst + pixel_offset) / sf0 - pixel_offset;
                    const int x0_src = (int) sycl::floor(x_src_f);
                    const float dx = x_src_f - (float) x0_src;

                    result = bicubic_interp(
                            bicubic_interp(load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src, -1, -1, ne00_src, ne01_src, i02_src, i03_src),
                                           load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src,  0, -1, ne00_src, ne01_src, i02_src, i03_src),
                                           load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src,  1, -1, ne00_src, ne01_src, i02_src, i03_src),
                                           load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src,  2, -1, ne00_src, ne01_src, i02_src, i03_src), dx),
                            bicubic_interp(load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src, -1,  0, ne00_src, ne01_src, i02_src, i03_src),
                                           load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src,  0,  0, ne00_src, ne01_src, i02_src, i03_src),
                                           load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src,  1,  0, ne00_src, ne01_src, i02_src, i03_src),
                                           load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src,  2,  0, ne00_src, ne01_src, i02_src, i03_src), dx),
                            bicubic_interp(load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src, -1,  1, ne00_src, ne01_src, i02_src, i03_src),
                                           load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src,  0,  1, ne00_src, ne01_src, i02_src, i03_src),
                                           load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src,  1,  1, ne00_src, ne01_src, i02_src, i03_src),
                                           load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src,  2,  1, ne00_src, ne01_src, i02_src, i03_src), dx),
                            bicubic_interp(load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src, -1,  2, ne00_src, ne01_src, i02_src, i03_src),
                                           load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src,  0,  2, ne00_src, ne01_src, i02_src, i03_src),
                                           load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src,  1,  2, ne00_src, ne01_src, i02_src, i03_src),
                                           load_clamped_f32(base, nb00, nb01, nb02, nb03, x0_src, y0_src,  2,  2, ne00_src, ne01_src, i02_src, i03_src), dx),
                            dy);
                }

                dst[index] = result;
            });
}
}

void ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int mode_flags = ggml_get_op_params_i32(dst, 0);
    const ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    float sf0 = (float) dst->ne[0] / src0->ne[0];
    float sf1 = (float) dst->ne[1] / src0->ne[1];
    const float sf2 = (float) dst->ne[2] / src0->ne[2];
    const float sf3 = (float) dst->ne[3] / src0->ne[3];

    float pixel_offset = 0.5f;
    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1 ? (float) (dst->ne[0] - 1) / (src0->ne[0] - 1) : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1 ? (float) (dst->ne[1] - 1) / (src0->ne[1] - 1) : sf1;
        pixel_offset = 0.0f;
    }

    ggml_sycl_upscale_mode sycl_mode = GGML_SYCL_UPSCALE_NEAREST;
    if (mode == GGML_SCALE_MODE_BILINEAR) {
        sycl_mode = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) ? GGML_SYCL_UPSCALE_BILINEAR_ANTIALIAS : GGML_SYCL_UPSCALE_BILINEAR;
    } else if (mode == GGML_SCALE_MODE_BICUBIC) {
        sycl_mode = GGML_SYCL_UPSCALE_BICUBIC;
    }

    upscale_dispatch(ctx.stream(), (const float *) src0->data, (float *) dst->data,
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            (int) src0->ne[0], (int) src0->ne[1], (int) dst->ne[0], (int) dst->ne[1], (int) dst->ne[2], (int) dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, sycl_mode);
}

bool ggml_sycl_supports_upscale(const ggml_tensor * op) {
    return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
}
