#ifndef GGML_SYCL_TOP_K_HPP
#define GGML_SYCL_TOP_K_HPP

#include "common.hpp"

void ggml_sycl_op_top_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_TOP_K_HPP
