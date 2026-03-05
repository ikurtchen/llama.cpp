#ifndef GGML_SYCL_CONV_HPP
#define GGML_SYCL_CONV_HPP

#include "common.hpp"

void ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst);

#endif // GGML_SYCL_CONV_HPP
