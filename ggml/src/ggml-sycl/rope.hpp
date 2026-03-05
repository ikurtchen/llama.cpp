#ifndef GGML_SYCL_ROPE_HPP
#define GGML_SYCL_ROPE_HPP

#include "common.hpp"

void ggml_sycl_rope(ggml_backend_sycl_context & ctx, ggml_tensor *dst);

#endif // GGML_SYCL_ROPE_HPP
