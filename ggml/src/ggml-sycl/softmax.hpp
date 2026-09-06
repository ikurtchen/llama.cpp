#pragma once

#include "common.hpp"

void ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_soft_max(const ggml_tensor * op);
bool ggml_sycl_supports_soft_max_back(const ggml_tensor * op);
