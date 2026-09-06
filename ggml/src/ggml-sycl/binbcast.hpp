#pragma once

#include "common.hpp"

void ggml_sycl_op_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

bool ggml_sycl_supports_binbcast(const ggml_tensor * op);
