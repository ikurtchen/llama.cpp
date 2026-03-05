#ifndef GGML_SYCL_CUMSUM_HPP
#define GGML_SYCL_CUMSUM_HPP

#include "common.hpp"

void ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_CUMSUM_HPP
