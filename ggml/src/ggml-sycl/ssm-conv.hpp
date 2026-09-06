#ifndef GGML_SYCL_SSM_CONV_HPP
#define GGML_SYCL_SSM_CONV_HPP

#include "common.hpp"

void ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_ssm_conv(const ggml_tensor * op);

#endif // GGML_SYCL_SSM_CONV_HPP
