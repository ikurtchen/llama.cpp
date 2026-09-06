#pragma once

#include "common.hpp"

void ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_clamp(const ggml_tensor * op);
