#ifndef GGML_SYCL_ACC_HPP
#define GGML_SYCL_ACC_HPP

#include "common.hpp"

void ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_acc(const ggml_tensor * op);

#endif // GGML_SYCL_ACC_HPP
