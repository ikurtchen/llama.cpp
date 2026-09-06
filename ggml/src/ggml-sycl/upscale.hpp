#ifndef GGML_SYCL_UPSCALE_HPP
#define GGML_SYCL_UPSCALE_HPP

#include "common.hpp"

void ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_upscale(const ggml_tensor * op);

#endif // GGML_SYCL_UPSCALE_HPP
