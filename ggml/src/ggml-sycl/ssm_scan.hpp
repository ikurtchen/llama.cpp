#ifndef GGML_SYCL_SSM_SCAN_HPP
#define GGML_SYCL_SSM_SCAN_HPP

#include "common.hpp"

void ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_SSM_SCAN_HPP
