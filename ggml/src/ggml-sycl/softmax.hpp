#ifndef GGML_SYCL_SOFTMAX_HPP
#define GGML_SYCL_SOFTMAX_HPP

#include "common.hpp"

#define SYCL_SOFT_MAX_BLOCK_SIZE 1024

void ggml_sycl_op_soft_max(ggml_backend_sycl_context &ctx, ggml_tensor *dst);

void ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_SOFTMAX_HPP
