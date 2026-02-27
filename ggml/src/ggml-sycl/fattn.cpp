//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "fattn.hpp"
#include "ggml-sycl.h"

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <sycl/half_type.hpp>

#define FATTN_KQ_STRIDE 256
#define SOFTMAX_FTZ_THRESHOLD -20.0f
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

static constexpr int ggml_sycl_fattn_vec_get_nthreads_device() {
    return 128;
}

static inline float get_alibi_slope(float max_bias, int64_t h, uint32_t n_head_log2, float m0, float m1) {
    if (max_bias == 0.0f) {
        return 0.0f;
    }
    const int64_t random = ~h;
    const float base = (random & n_head_log2) ? m1 : m0;
    const int64_t exph = random >> uint32_t(std::floor(std::log2(float(n_head_log2))));
    return base * std::pow(2.0f, float(n_head_log2 - exph - 1));
}

static inline float half_to_float(sycl::half val) {
    return sycl::vec<sycl::half, 1>(val).convert<float, sycl::rounding_mode::automatic>()[0];
}

static inline sycl::half float_to_half(float val) {
    return sycl::vec<float, 1>(val).convert<sycl::half, sycl::rounding_mode::rte>()[0];
}

static inline sycl::half2 make_half2(float x, float y) {
    sycl::half2 result;
    result.x() = float_to_half(x);
    result.y() = float_to_half(y);
    return result;
}

static inline sycl::float2 make_float2(float x, float y) {
    return {x, y};
}

static inline sycl::half2 hadd2(sycl::half2 a, sycl::half2 b) {
    return {sycl::half(float(a.x()) + float(b.x())), sycl::half(float(a.y()) + float(b.y()))};
}

static inline sycl::half2 hmul2(sycl::half2 a, sycl::half2 b) {
    return {sycl::half(float(a.x()) * float(b.x())), sycl::half(float(a.y()) * float(b.y()))};
}

bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (Q->ne[0] != K->ne[0] || Q->ne[0] != V->ne[0]) {
        return false;
    }

    if (K->ne[0] > 256) {
        return false;
    }

    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false;
    }

    return true;
}

template<int D, int ncols>
static void ggml_sycl_flash_attn_ext_vec_f16(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const float * src0 = (const float *) Q->data;
    const float * src1 = (const float *) K->data;
    const float * src2 = (const float *) V->data;
    const float * src3 = mask ? (const float *) mask->data : nullptr;

    const int32_t ne00 = Q->ne[0];
    const int32_t ne01 = Q->ne[1];
    const int32_t ne02 = Q->ne[2];
    const int32_t ne03 = Q->ne[3];

    const int32_t nb01 = Q->nb[1];
    const int32_t nb02 = Q->nb[2];
    const int32_t nb03 = Q->nb[3];

    const int32_t ne10 = K->ne[0];
    const int32_t ne11 = K->ne[1];
    const int32_t ne12 = K->ne[2];
    const int32_t ne13 = K->ne[3];

    const int32_t nb11 = K->nb[1];
    const int32_t nb12 = K->nb[2];
    const int64_t nb13 = K->nb[3];

    const int32_t nb21 = V->nb[1];
    const int32_t nb22 = V->nb[2];
    const int64_t nb23 = V->nb[3];

    const int32_t ne31 = mask ? mask->ne[1] : 1;
    const int32_t ne32 = mask ? mask->ne[2] : 1;
    const int32_t ne33 = mask ? mask->ne[3] : 1;
    const int32_t nb31 = mask ? mask->nb[1] : 0;
    const int32_t nb32 = mask ? mask->nb[2] : 0;
    const int64_t nb33 = mask ? mask->nb[3] : 0;

    float * dst_float = (float *) dst->data;
    float2 * dst_meta = nullptr;

    const float scale = 1.0f / sqrtf((float) ne00);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    float m0 = 1.0f;
    float m1 = 1.0f;
    memcpy(&m0, (const float *) dst->op_params + 3, sizeof(float));
    memcpy(&m1, (const float *) dst->op_params + 4, sizeof(float));

    uint32_t n_head_log2 = 1;
    while (n_head_log2 < ne02) {
        n_head_log2 *= 2;
    }
    n_head_log2 /= 2;

    const int gqa_ratio = ne02 / ne12;

    const int nthreads = ggml_sycl_fattn_vec_get_nthreads_device();
    constexpr int nwarps = nthreads / WARP_SIZE;
    constexpr int nthreads_KQ = WARP_SIZE;
    constexpr int nthreads_V = WARP_SIZE;

    constexpr int ne_KQ = ncols * D;
    constexpr int V_cols_per_iter = WARP_SIZE / nthreads_V;
    constexpr int ne_combine = nwarps * V_cols_per_iter * D;
    constexpr int KQ_size = ne_KQ > ne_combine ? ne_KQ : ne_combine;

    const int32_t nrows_per_block = ncols;
    const int32_t nblocks_x = (ne01 + nrows_per_block - 1) / nrows_per_block;
    const int32_t nblocks_y = (ne11 + nthreads - 1) / nthreads;
    const int32_t nblocks_z = ne02 * ne03;

    const size_t kq_size = KQ_size * sizeof(float);

    sycl::range<3> block_dims(1, nwarps, WARP_SIZE);
    sycl::range<3> global_size(1, nblocks_y * nwarps, nblocks_x * WARP_SIZE);

    const int ncols_ = ncols;
    const int D_ = D;
    const float scale_ = scale;
    const float max_bias_ = max_bias;
    const float m0_ = m0;
    const float m1_ = m1;
    const int ne01_ = ne01;
    const int ne02_ = ne02;
    const int ne11_ = ne11;
    const int nb01_ = nb01;
    const int nb02_ = nb02;
    const int nb03_ = nb03;
    const int nb11_ = nb11;
    const int nb12_ = nb12;
    const int nb13_ = nb13;
    const int nb21_ = nb21;
    const int nb22_ = nb22;
    const int nb23_ = nb23;
    const int nb31_ = nb31;
    const int nb33_ = nb33;
    const int ne31_ = ne31;
    const int ne33_ = ne33;
    const int gqa_ratio_ = gqa_ratio;
    const int nblocks_y_ = nblocks_y;
    const int nthreads_ = nthreads;
    const int KQ_size_ = KQ_size;

    ctx.queue().submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> KQ_acc(sycl::range<1>(KQ_size_), cgh);
        sycl::local_accessor<float, 2> KQ_max_shared(sycl::range<1>(ncols_), cgh);
        sycl::local_accessor<float, 2> KQ_sum_shared(sycl::range<1>(ncols_), cgh);

        cgh.parallel_for(sycl::nd_range<3>(global_size, block_dims), [=](sycl::nd_item<3> item_ct1)
            [[intel::reqd_sub_group_size(WARP_SIZE)]] {
            auto & item_ct = item_ct1;
            const int tid = item_ct.get_local_id(2);
            const int warp_id = item_ct.get_local_id(1);
            const int lane_id = tid;

            const int ic0 = item_ct.get_group(2) * ncols_;

            const int sequence = item_ct.get_group(0) / ne02_;
            const int head = item_ct.get_group(0) - sequence * ne02_;

            const float slope = get_alibi_slope(max_bias_, head, ne02_, m0_, m1_);

            const char * Q_ptr = (const char *) src0 + nb03_ * sequence + nb02_ * head;
            const char * K_ptr = (const char *) src1 + nb13_ * sequence + nb12_ * (head / gqa_ratio_);
            const char * V_ptr = (const char *) src2 + nb23_ * sequence + nb22_ * (head / gqa_ratio_);
            const char * mask_ptr = src3 ? (const char *) src3 + nb33_ * (sequence % ne33_) + nb31_ * ic0 : nullptr;

            float KQ_max_local = -FLT_MAX / 2.0f;
            float KQ_sum_local = 0.0f;
            sycl::half2 VKQ[1][D_ / 2 / nthreads_V];
            for (int i = 0; i < D_ / 2 / nthreads_V; ++i) {
                VKQ[0][i] = {sycl::half(0.0f), sycl::half(0.0f)};
            }

            sycl::half2 Q_reg[ncols_][D_ / 2 / nthreads_KQ];
            for (int j = 0; j < ncols_; ++j) {
                for (int i = 0; i < D_ / 2 / nthreads_KQ; ++i) {
                    Q_reg[j][i] = {sycl::half(0.0f), sycl::half(0.0f)};
                }
            }

            const sycl::half2 scale_h2 = make_half2(scale_, scale_);

            for (int j = 0; j < ncols_; ++j) {
                if (ncols_ == 1 || ic0 + j < ne01_) {
                    const sycl::half2 * Q_j = (const sycl::half2 *) (Q_ptr + j * nb01_);
                    for (int i0 = 0; i0 < D_ / 2; i0 += nthreads_KQ) {
                        const int i = i0 + lane_id;
                        if (i < D_ / 2) {
                            Q_reg[j][i0 / nthreads_KQ] = Q_j[i] * scale_h2;
                        }
                    }
                }
            }

            const int k_VKQ_max = ne11_;
            const int k_start = item_ct.get_group(1) * nthreads_;

            for (int k_VKQ_0 = k_start; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += nblocks_y_ * nthreads_) {
                const int k = k_VKQ_0 + warp_id * WARP_SIZE + lane_id;
                const bool k_valid = k < k_VKQ_max;

                const char * K_row = k_valid ? K_ptr + k * nb11_ : K_ptr;
                const char * V_row = k_valid ? V_ptr + k * nb21_ : V_ptr;
                const float * mask_row = (mask_ptr && k_valid) ? (const float *) (mask_ptr + k) : nullptr;

                float KQ_max_new = KQ_max_local;

                float KQ_local = 0.0f;
                if (k_valid) {
                    for (int j = 0; j < ncols_; ++j) {
                        float sum = 0.0f;
                        const sycl::half2 * K_h2 = (const sycl::half2 *) K_row;
                        for (int i0 = 0; i0 < D_ / 2; ++i0) {
                            sum += float(Q_reg[j][i0].x()) * float(K_h2[i0].x()) +
                                   float(Q_reg[j][i0].y()) * float(K_h2[i0].y());
                        }

                        if (mask_row && (ncols_ == 1 || ic0 + j < ne01_)) {
                            const float mask_val = mask_row[j * ne11_];
                            sum += slope * mask_val;
                        }

                        KQ_max_new = fmaxf(KQ_max_new, sum + FATTN_KQ_MAX_OFFSET);
                        KQ_local = sum;
                    }
                }

                auto subgrp = item_ct.get_sub_group();
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    KQ_max_new = fmaxf(KQ_max_new, sycl::permute_sub_group_by_xor(subgrp, KQ_max_new, offset));
                }

                const float KQ_max_scale = expf(KQ_max_local - KQ_max_new);
                KQ_max_local = KQ_max_new;

                KQ_local = expf(KQ_local - KQ_max_local);
                KQ_sum_local = KQ_sum_local * KQ_max_scale + KQ_local;

                const int kq_offset = warp_id * WARP_SIZE + lane_id;
                if (k_valid && kq_offset < KQ_size_) {
                    KQ_acc[kq_offset] = KQ_local;
                }

                for (int i = 0; i < D_ / 2 / nthreads_V; ++i) {
                    VKQ[0][i] = sycl::half2(
                        sycl::half(float(VKQ[0][i].x()) * KQ_max_scale),
                        sycl::half(float(VKQ[0][i].y()) * KQ_max_scale)
                    );
                }

                item_ct.barrier(sycl::local_memory_fence);

                if (k_valid) {
                    for (int j = 0; j < ncols_; ++j) {
                        for (int i0 = 0; i0 < D_ / 2; i0 += WARP_SIZE) {
                            const int i = i0 + lane_id;
                            if (i < D_ / 2) {
                                const float KQ_k = KQ_acc[j * nthreads_ + i];
                                const sycl::half2 * V_h2 = (const sycl::half2 *) V_row;
                                VKQ[0][i0 / WARP_SIZE] = hadd2(VKQ[0][i0 / WARP_SIZE],
                                    hmul2(V_h2[i], sycl::half2(sycl::half(KQ_k), sycl::half(KQ_k))));
                            }
                        }
                    }
                }

                item_ct.barrier(sycl::local_memory_fence);
            }

            if (warp_id == 0 && lane_id < ncols_) {
                KQ_max_shared[lane_id][0] = KQ_max_local;
                KQ_sum_shared[lane_id][0] = KQ_sum_local;
            }

            item_ct.barrier(sycl::local_memory_fence);

            auto subgrp = item_ct.get_sub_group();

            for (int j = 0; j < ncols_; ++j) {
                if (ncols_ > 1 && ic0 + j >= ne01_) {
                    continue;
                }

                float kqmax_new = KQ_max_shared[j][0];
                kqmax_new = fmaxf(kqmax_new, sycl::reduce_over_group(subgrp, kqmax_new, sycl::plus<float>()));
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    kqmax_new = fmaxf(kqmax_new, sycl::permute_sub_group_by_xor(subgrp, kqmax_new, offset));
                }

                const float kqmax_scale = expf(KQ_max_shared[j][0] - kqmax_new);
                KQ_max_shared[j][0] = kqmax_new;

                for (int i = 0; i < D_ / 2 / nthreads_V; ++i) {
                    VKQ[0][i] = sycl::half2(
                        sycl::half(float(VKQ[0][i].x()) * kqmax_scale),
                        sycl::half(float(VKQ[0][i].y()) * kqmax_scale)
                    );
                }

                KQ_sum_local = KQ_sum_shared[j][0] * kqmax_scale;
                float kq_sum_new = KQ_sum_local + sycl::reduce_over_group(subgrp, KQ_sum_local, sycl::plus<float>());
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    kq_sum_new += sycl::permute_sub_group_by_xor(subgrp, kq_sum_new, offset);
                }

                if (lane_id == 0) {
                    KQ_sum_shared[j][0] = kq_sum_new;
                }

                item_ct.barrier(sycl::local_memory_fence);

                if (kq_sum_new > 0.0f) {
                    for (int i0 = 0; i0 < D_; i0 += nthreads_) {
                        const int i = i0 + lane_id;
                        if (i < D_) {
                            float dst_val = 0.0f;
                            for (int w = 0; w < nwarps; ++w) {
                                const float vkq_val = float(VKQ[0][i / 2].x());
                                dst_val += vkq_val;
                            }
                            dst_val /= kq_sum_new;

                            const size_t dst_idx = ((sequence * ne01_ + ic0 + j) * ne02_ + head) * D_ + i;
                            dst_float[dst_idx] = dst_val;
                        }
                    }
                }

                if (j < ncols_ - 1) {
                    item_ct.barrier(sycl::local_memory_fence);
                }
            }
        });
    });
}

void ggml_sycl_op_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const int D = Q->ne[0];
    const int ncols = Q->ne[1];

    if (D == 128 && ncols == 1) {
        ggml_sycl_flash_attn_ext_vec_f16<128, 1>(ctx, dst);
    } else if (D == 128 && ncols == 2) {
        ggml_sycl_flash_attn_ext_vec_f16<128, 2>(ctx, dst);
    } else if (D == 64 && ncols == 1) {
        ggml_sycl_flash_attn_ext_vec_f16<64, 1>(ctx, dst);
    } else if (D == 64 && ncols == 2) {
        ggml_sycl_flash_attn_ext_vec_f16<64, 2>(ctx, dst);
    } else if (D == 256 && ncols == 1) {
        ggml_sycl_flash_attn_ext_vec_f16<256, 1>(ctx, dst);
    } else if (D == 256 && ncols == 2) {
        ggml_sycl_flash_attn_ext_vec_f16<256, 2>(ctx, dst);
    } else if (D == 40 && ncols == 1) {
        ggml_sycl_flash_attn_ext_vec_f16<40, 1>(ctx, dst);
    } else if (D == 40 && ncols == 2) {
        ggml_sycl_flash_attn_ext_vec_f16<40, 2>(ctx, dst);
    } else {
        GGML_UNUSED(ctx);
        GGML_UNUSED(dst);
    }
}
