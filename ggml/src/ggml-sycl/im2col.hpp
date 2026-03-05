#ifndef GGML_SYCL_IM2COL_HPP
#define GGML_SYCL_IM2COL_HPP

#include "common.hpp"

void ggml_sycl_op_im2col(
        ggml_backend_sycl_context & ctx, ggml_tensor *dst);

void ggml_sycl_op_im2col_3d(
        ggml_backend_sycl_context & ctx, ggml_tensor *dst);

#endif // GGML_SYCL_IM2COL_HPP
