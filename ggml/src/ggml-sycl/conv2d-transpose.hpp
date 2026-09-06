#ifndef GGML_SYCL_CONV2D_TRANSPOSE_HPP
#define GGML_SYCL_CONV2D_TRANSPOSE_HPP

#include "common.hpp"

void ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_conv2d_transpose(const ggml_tensor * op);

#endif // GGML_SYCL_CONV2D_TRANSPOSE_HPP
