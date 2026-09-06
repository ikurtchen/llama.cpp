#pragma once

#include "common.hpp"

void ggml_sycl_op_count_equal(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_count_equal(const ggml_tensor * op);
