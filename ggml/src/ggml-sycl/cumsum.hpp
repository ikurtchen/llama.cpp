#ifndef GGML_SYCL_CUMSUM_HPP
#define GGML_SYCL_CUMSUM_HPP

#include "common.hpp"

void ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_cumsum(const ggml_tensor * op);

#endif // GGML_SYCL_CUMSUM_HPP
