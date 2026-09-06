#ifndef GGML_SYCL_OUT_PROD_HPP
#define GGML_SYCL_OUT_PROD_HPP

#include "common.hpp"

void ggml_sycl_out_prod(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_out_prod(const ggml_tensor * op);

#endif // GGML_SYCL_OUT_PROD_HPP
