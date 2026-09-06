#pragma once

#include "common.hpp"

void ggml_sycl_cpy(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, ggml_tensor * dst);
bool ggml_sycl_supports_cpy(const ggml_tensor * op);
