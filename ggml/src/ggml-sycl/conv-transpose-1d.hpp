#ifndef GGML_SYCL_CONV_TRANSPOSE_1D_HPP
#define GGML_SYCL_CONV_TRANSPOSE_1D_HPP

#include "common.hpp"

void ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_conv_transpose_1d(const ggml_tensor * op);

#endif // GGML_SYCL_CONV_TRANSPOSE_1D_HPP
