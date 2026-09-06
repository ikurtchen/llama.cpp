#ifndef GGML_SYCL_ARANGE_HPP
#define GGML_SYCL_ARANGE_HPP

#include "common.hpp"

void ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_arange(const ggml_tensor * op);

#endif // GGML_SYCL_ARANGE_HPP
