#ifndef GGML_SYCL_WKV_HPP
#define GGML_SYCL_WKV_HPP

#include "common.hpp"

void ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_rwkv_wkv6(const ggml_tensor * op);
bool ggml_sycl_supports_rwkv_wkv7(const ggml_tensor * op);

#endif // GGML_SYCL_WKV_HPP
