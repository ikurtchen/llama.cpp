#ifndef GGML_SYCL_POOL2D_HPP
#define GGML_SYCL_POOL2D_HPP

#include "common.hpp"

void ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_pool2d(const ggml_tensor * op);

#endif // GGML_SYCL_POOL2D_HPP
