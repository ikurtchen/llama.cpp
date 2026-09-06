#ifndef GGML_SYCL_ROLL_HPP
#define GGML_SYCL_ROLL_HPP

#include "common.hpp"

void ggml_sycl_op_roll(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_roll(const ggml_tensor * op);

#endif // GGML_SYCL_ROLL_HPP
