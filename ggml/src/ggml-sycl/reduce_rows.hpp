#ifndef GGML_SYCL_REDUCE_ROWS_HPP
#define GGML_SYCL_REDUCE_ROWS_HPP

#include "common.hpp"

void ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

bool ggml_sycl_supports_sum(const ggml_tensor * op);
bool ggml_sycl_supports_sum_rows(const ggml_tensor * op);
bool ggml_sycl_supports_mean(const ggml_tensor * op);

#endif // GGML_SYCL_REDUCE_ROWS_HPP
