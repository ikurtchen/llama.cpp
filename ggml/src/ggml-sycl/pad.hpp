#ifndef GGML_SYCL_PAD_HPP
#define GGML_SYCL_PAD_HPP

#include "common.hpp"

void ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_pad(const ggml_tensor * op);

#endif // GGML_SYCL_PAD_HPP
