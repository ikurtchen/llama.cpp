#pragma once

#include "common.hpp"

void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_get_rows(const ggml_tensor * op);
