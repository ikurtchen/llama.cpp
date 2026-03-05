#ifndef GGML_SYCL_NORM_HPP
#define GGML_SYCL_NORM_HPP

#include "common.hpp"

void ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#endif // GGML_SYCL_NORM_HPP
