#ifndef GGML_SYCL_CONCAT_HPP
#define GGML_SYCL_CONCAT_HPP

#include "common.hpp"

void ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst);

#endif // GGML_SYCL_CONCAT_HPP
