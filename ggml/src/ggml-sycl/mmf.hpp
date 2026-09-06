#pragma once

#include "common.hpp"

void ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_mul_mat(const ggml_tensor * op);
