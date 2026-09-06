#ifndef GGML_SYCL_GLA_HPP
#define GGML_SYCL_GLA_HPP

#include "common.hpp"

void ggml_sycl_op_gated_linear_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_gated_linear_attn(const ggml_tensor * op);

#endif // GGML_SYCL_GLA_HPP
