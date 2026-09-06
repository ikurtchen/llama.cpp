#ifndef GGML_SYCL_ADD_ID_HPP
#define GGML_SYCL_ADD_ID_HPP

#include "common.hpp"

void ggml_sycl_op_add_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_add_id(const ggml_tensor * op);

#endif // GGML_SYCL_ADD_ID_HPP
