#ifndef GGML_SYCL_FILL_HPP
#define GGML_SYCL_FILL_HPP

#include "common.hpp"

void ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_fill(const ggml_tensor * op);

#endif // GGML_SYCL_FILL_HPP
