#ifndef GGML_SYCL_PAD_HPP
#define GGML_SYCL_PAD_HPP

#include "common.hpp"

#define SYCL_PAD_BLOCK_SIZE 256

void ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_PAD_HPP
