#pragma once

#include "common.hpp"

void ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_scale(const ggml_tensor * op);
