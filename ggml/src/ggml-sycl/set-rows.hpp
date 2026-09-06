#ifndef GGML_SYCL_SET_ROWS_HPP
#define GGML_SYCL_SET_ROWS_HPP

#include "common.hpp"

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_set_rows(const ggml_tensor * op);

#endif // GGML_SYCL_SET_ROWS_HPP
