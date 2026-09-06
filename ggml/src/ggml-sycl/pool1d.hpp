#ifndef GGML_SYCL_POOL1D_HPP
#define GGML_SYCL_POOL1D_HPP

#include "common.hpp"

void ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_pool1d(const ggml_tensor * op);

#endif // GGML_SYCL_POOL1D_HPP
