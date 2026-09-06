#pragma once

#include "common.hpp"

void ggml_sycl_op_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

bool ggml_sycl_supports_norm(const ggml_tensor * op);
