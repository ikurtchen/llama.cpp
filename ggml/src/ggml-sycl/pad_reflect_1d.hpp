#ifndef GGML_SYCL_PAD_REFLECT_1D_HPP
#define GGML_SYCL_PAD_REFLECT_1D_HPP

#include "common.hpp"

void ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_pad_reflect_1d(const ggml_tensor * op);

#endif // GGML_SYCL_PAD_REFLECT_1D_HPP
