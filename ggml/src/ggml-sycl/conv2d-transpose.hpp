#ifndef GGML_SYCL_CONV2D_TRANSPOSE_HPP
#define GGML_SYCL_CONV2D_TRANSPOSE_HPP

#include "common.hpp"

void ggml_sycl_op_conv_transpose_2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_CONV2D_TRANSPOSE_HPP
