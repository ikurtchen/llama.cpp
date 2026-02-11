#ifndef GGML_SYCL_CONV2D_HPP
#define GGML_SYCL_CONV2D_HPP

#include "common.hpp"

void ggml_sycl_op_conv_2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_CONV2D_HPP
