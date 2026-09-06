#pragma once

#include "common.hpp"

void ggml_sycl_op_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_flash_attn_ext(const ggml_tensor * op);
