#ifndef GGML_SYCL_ARGMAX_HPP
#define GGML_SYCL_ARGMAX_HPP

#include "common.hpp"

void ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_argmax(const ggml_tensor * op);

#endif // GGML_SYCL_ARGMAX_HPP
