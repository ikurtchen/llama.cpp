#pragma once

#include "common.hpp"

// Plain unary ops (dst->op == GGML_OP_UNARY, sub-op via ggml_get_unary_op).
void ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_unary(const ggml_tensor * op);

// Gated (GLU) ops (dst->op == GGML_OP_GLU, sub-op via ggml_get_glu_op).
void ggml_sycl_op_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_glu(const ggml_tensor * op);

// SQR/SQRT/SIN/COS/LOG are their own top-level GGML_OP_* (not GGML_OP_UNARY sub-ops upstream).
void ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_unary_f32f16(const ggml_tensor * op);
