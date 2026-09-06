#ifndef GGML_SYCL_SSM_SCAN_HPP
#define GGML_SYCL_SSM_SCAN_HPP

#include "common.hpp"

void ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_ssm_scan(const ggml_tensor * op);

#endif // GGML_SYCL_SSM_SCAN_HPP
