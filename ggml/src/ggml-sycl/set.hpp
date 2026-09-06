#ifndef GGML_SYCL_SET_HPP
#define GGML_SYCL_SET_HPP

#include "common.hpp"

void ggml_sycl_op_set(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_set(const ggml_tensor * op);

#endif // GGML_SYCL_SET_HPP
