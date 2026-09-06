#pragma once

#include "common.hpp"

#define GGML_COMMON_DECL_SYCL
#include "ggml-common.h"

#include <cstdint>
#include <cstring>

template <typename dst_t>
inline dst_t ggml_sycl_cast_f32(float v);

template <>
inline float ggml_sycl_cast_f32<float>(float v) {
    return v;
}

template <>
inline sycl::half ggml_sycl_cast_f32<sycl::half>(float v) {
    return (sycl::half) v;
}

inline void ggml_sycl_get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

template <typename dst_t>
inline dst_t ggml_sycl_dequantize_q4_0_at(const block_q4_0 * x, int64_t i) {
    const int64_t ib = i / QK4_0;
    const int      l = i % QK4_0;
    const uint8_t  q = x[ib].qs[l % (QK4_0 / 2)];
    const int      v = l < QK4_0 / 2 ? (q & 0x0F) - 8 : (q >> 4) - 8;
    return ggml_sycl_cast_f32<dst_t>((float) v * (float) x[ib].d);
}

template <typename dst_t>
inline dst_t ggml_sycl_dequantize_q4_1_at(const block_q4_1 * x, int64_t i) {
    const int64_t ib = i / QK4_1;
    const int      l = i % QK4_1;
    const uint8_t  q = x[ib].qs[l % (QK4_1 / 2)];
    const int      v = l < QK4_1 / 2 ? (q & 0x0F) : (q >> 4);
    const float    d = (float) x[ib].data.d;
    const float    m = (float) x[ib].data.m;
    return ggml_sycl_cast_f32<dst_t>((float) v * d + m);
}

template <typename dst_t>
inline dst_t ggml_sycl_dequantize_q5_0_at(const block_q5_0 * x, int64_t i) {
    const int64_t ib = i / QK5_0;
    const int      l = i % QK5_0;
    uint32_t       qh = 0;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int      j = l % (QK5_0 / 2);
    const uint8_t  q = x[ib].qs[j];
    const int      hi = (qh >> l) & 1;
    const int      v = (l < QK5_0 / 2 ? (q & 0x0F) : (q >> 4)) | (hi << 4);
    return ggml_sycl_cast_f32<dst_t>(((float) v - 16.0f) * (float) x[ib].d);
}

template <typename dst_t>
inline dst_t ggml_sycl_dequantize_q5_1_at(const block_q5_1 * x, int64_t i) {
    const int64_t ib = i / QK5_1;
    const int      l = i % QK5_1;
    uint32_t       qh = 0;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int      j = l % (QK5_1 / 2);
    const uint8_t  q = x[ib].qs[j];
    const int      hi = (qh >> l) & 1;
    const int      v = (l < QK5_1 / 2 ? (q & 0x0F) : (q >> 4)) | (hi << 4);
    const float    d = (float) x[ib].data.d;
    const float    m = (float) x[ib].data.m;
    return ggml_sycl_cast_f32<dst_t>((float) v * d + m);
}

template <typename dst_t>
inline dst_t ggml_sycl_dequantize_q8_0_at(const block_q8_0 * x, int64_t i) {
    const int64_t ib = i / QK8_0;
    const int      l = i % QK8_0;
    return ggml_sycl_cast_f32<dst_t>((float) x[ib].qs[l] * (float) x[ib].d);
}

template <typename dst_t>
inline dst_t ggml_sycl_dequantize_q2_K_at(const block_q2_K * x, int64_t i) {
    const int64_t ib = i / QK_K;
    const int      l = i % QK_K;
    const int      n = l / 128;
    const int      r = l % 128;
    const int      j = r / 32;
    const int      p = (r % 32) / 16;
    const int      m = r % 16;
    const int      is = 8 * n + 2 * j + p;
    const int      shift = 2 * j;
    const uint8_t  sc = x[ib].scales[is];
    const float    d = (float) x[ib].data.d;
    const float    dmin = (float) x[ib].data.dmin;
    const float    dl = d * (sc & 0x0F);
    const float    ml = dmin * (sc >> 4);
    const uint8_t  q = x[ib].qs[32 * n + 16 * p + m];
    return ggml_sycl_cast_f32<dst_t>(dl * ((q >> shift) & 0x03) - ml);
}

inline uint8_t ggml_sycl_q3_K_scale(const block_q3_K & x, int is) {
    if (is < 4) {
        return (x.scales[is - 0] & 0x0F) | (((x.scales[is + 8] >> 0) & 3) << 4);
    }
    if (is < 8) {
        return (x.scales[is - 0] & 0x0F) | (((x.scales[is + 4] >> 2) & 3) << 4);
    }
    if (is < 12) {
        return (x.scales[is - 8] >> 4) | (((x.scales[is + 0] >> 4) & 3) << 4);
    }
    return (x.scales[is - 8] >> 4) | (((x.scales[is - 4] >> 6) & 3) << 4);
}

template <typename dst_t>
inline dst_t ggml_sycl_dequantize_q3_K_at(const block_q3_K * x, int64_t i) {
    const int64_t ib = i / QK_K;
    const int      l = i % QK_K;
    const int      n = l / 128;
    const int      r = l % 128;
    const int      j = r / 32;
    const int      m = r % 32;
    const int      is = 8 * n + 2 * j + m / 16;
    const uint8_t  q = x[ib].qs[32 * n + m];
    const uint8_t  h = x[ib].hmask[m];
    const uint8_t  mask = (uint8_t) (1u << (4 * n + j));
    const int      v = ((q >> (2 * j)) & 0x03) - ((h & mask) ? 0 : 4);
    const float    d = (float) x[ib].d;
    const float    dl = d * ((int) ggml_sycl_q3_K_scale(x[ib], is) - 32);
    return ggml_sycl_cast_f32<dst_t>(dl * v);
}

template <typename dst_t>
inline dst_t ggml_sycl_dequantize_q4_K_at(const block_q4_K * x, int64_t i) {
    const int64_t ib = i / QK_K;
    const int      l = i % QK_K;
    const int      chunk = l / 64;
    const int      lane = l % 32;
    const int      half = (l % 64) / 32;
    uint8_t        sc = 0;
    uint8_t        m = 0;
    ggml_sycl_get_scale_min_k4(2 * chunk + half, x[ib].scales, sc, m);
    const uint8_t  q = x[ib].qs[32 * chunk + lane];
    const int      v = half == 0 ? (q & 0x0F) : (q >> 4);
    const float    d = (float) x[ib].data.d;
    const float    dmin = (float) x[ib].data.dmin;
    return ggml_sycl_cast_f32<dst_t>(d * sc * v - dmin * m);
}

template <typename dst_t>
inline dst_t ggml_sycl_dequantize_q5_K_at(const block_q5_K * x, int64_t i) {
    const int64_t ib = i / QK_K;
    const int      l = i % QK_K;
    const int      chunk = l / 64;
    const int      lane = l % 32;
    const int      half = (l % 64) / 32;
    uint8_t        sc = 0;
    uint8_t        m = 0;
    ggml_sycl_get_scale_min_k4(2 * chunk + half, x[ib].scales, sc, m);
    const uint8_t  ql = x[ib].qs[32 * chunk + lane];
    const uint8_t  qh = x[ib].qh[lane];
    const uint8_t  hm = (uint8_t) (1u << (2 * chunk + half));
    const int      v = (half == 0 ? (ql & 0x0F) : (ql >> 4)) + ((qh & hm) ? 16 : 0);
    const float    d = (float) x[ib].data.d;
    const float    dmin = (float) x[ib].data.dmin;
    return ggml_sycl_cast_f32<dst_t>(d * sc * v - dmin * m);
}

template <typename dst_t>
inline dst_t ggml_sycl_dequantize_q6_K_at(const block_q6_K * x, int64_t i) {
    const int64_t ib = i / QK_K;
    const int      l = i % QK_K;
    const int      n = l / 128;
    const int      r = l % 128;
    const int      quarter = r / 32;
    const int      lane = r % 32;
    const uint8_t * ql = x[ib].ql + 64 * n;
    const uint8_t * qh = x[ib].qh + 32 * n;
    const int8_t  * sc = x[ib].scales + 8 * n;
    const int      is = lane / 16;

    int8_t q = 0;
    int8_t s = 0;
    switch (quarter) {
        case 0:
            q = (int8_t) ((ql[lane + 0] & 0x0F) | (((qh[lane] >> 0) & 0x03) << 4)) - 32;
            s = sc[is + 0];
            break;
        case 1:
            q = (int8_t) ((ql[lane + 32] & 0x0F) | (((qh[lane] >> 2) & 0x03) << 4)) - 32;
            s = sc[is + 2];
            break;
        case 2:
            q = (int8_t) ((ql[lane + 0] >> 4) | (((qh[lane] >> 4) & 0x03) << 4)) - 32;
            s = sc[is + 4];
            break;
        default:
            q = (int8_t) ((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 0x03) << 4)) - 32;
            s = sc[is + 6];
            break;
    }

    return ggml_sycl_cast_f32<dst_t>((float) x[ib].d * (float) s * (float) q);
}

template <typename dst_t>
inline void ggml_sycl_dequantize_q4_0_block(const block_q4_0 & x, dst_t * y) {
    for (int i = 0; i < QK4_0; ++i) {
        y[i] = ggml_sycl_dequantize_q4_0_at<dst_t>(&x, i);
    }
}

template <typename dst_t>
inline void ggml_sycl_dequantize_q4_1_block(const block_q4_1 & x, dst_t * y) {
    for (int i = 0; i < QK4_1; ++i) {
        y[i] = ggml_sycl_dequantize_q4_1_at<dst_t>(&x, i);
    }
}

template <typename dst_t>
inline void ggml_sycl_dequantize_q5_0_block(const block_q5_0 & x, dst_t * y) {
    for (int i = 0; i < QK5_0; ++i) {
        y[i] = ggml_sycl_dequantize_q5_0_at<dst_t>(&x, i);
    }
}

template <typename dst_t>
inline void ggml_sycl_dequantize_q5_1_block(const block_q5_1 & x, dst_t * y) {
    for (int i = 0; i < QK5_1; ++i) {
        y[i] = ggml_sycl_dequantize_q5_1_at<dst_t>(&x, i);
    }
}

template <typename dst_t>
inline void ggml_sycl_dequantize_q8_0_block(const block_q8_0 & x, dst_t * y) {
    for (int i = 0; i < QK8_0; ++i) {
        y[i] = ggml_sycl_dequantize_q8_0_at<dst_t>(&x, i);
    }
}

template <typename dst_t>
inline void ggml_sycl_dequantize_q2_K_block(const block_q2_K & x, dst_t * y) {
    for (int i = 0; i < QK_K; ++i) {
        y[i] = ggml_sycl_dequantize_q2_K_at<dst_t>(&x, i);
    }
}

template <typename dst_t>
inline void ggml_sycl_dequantize_q3_K_block(const block_q3_K & x, dst_t * y) {
    for (int i = 0; i < QK_K; ++i) {
        y[i] = ggml_sycl_dequantize_q3_K_at<dst_t>(&x, i);
    }
}

template <typename dst_t>
inline void ggml_sycl_dequantize_q4_K_block(const block_q4_K & x, dst_t * y) {
    for (int i = 0; i < QK_K; ++i) {
        y[i] = ggml_sycl_dequantize_q4_K_at<dst_t>(&x, i);
    }
}

template <typename dst_t>
inline void ggml_sycl_dequantize_q5_K_block(const block_q5_K & x, dst_t * y) {
    for (int i = 0; i < QK_K; ++i) {
        y[i] = ggml_sycl_dequantize_q5_K_at<dst_t>(&x, i);
    }
}

template <typename dst_t>
inline void ggml_sycl_dequantize_q6_K_block(const block_q6_K & x, dst_t * y) {
    for (int i = 0; i < QK_K; ++i) {
        y[i] = ggml_sycl_dequantize_q6_K_at<dst_t>(&x, i);
    }
}

template <typename T>
using to_t_sycl_t = void (*)(const void * x, T * y, int64_t k, sycl::queue & q);

using to_fp32_sycl_t = to_t_sycl_t<float>;
using to_fp16_sycl_t = to_t_sycl_t<sycl::half>;

void ggml_sycl_convert_to_fp32(sycl::queue & q, const ggml_tensor * src, float * dst);
void ggml_sycl_convert_to_fp16(sycl::queue & q, const ggml_tensor * src, sycl::half * dst);

to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type);
to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type);
