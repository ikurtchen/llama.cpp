#ifndef GGML_SYCL_DSV4_HC_HPP
#define GGML_SYCL_DSV4_HC_HPP

#include "common.hpp"

void ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

bool ggml_sycl_supports_dsv4_hc_comb(const ggml_tensor * op);
bool ggml_sycl_supports_dsv4_hc_pre(const ggml_tensor * op);
bool ggml_sycl_supports_dsv4_hc_post(const ggml_tensor * op);

#endif // GGML_SYCL_DSV4_HC_HPP
