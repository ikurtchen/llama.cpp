#ifndef GGML_SYCL_DIAG_HPP
#define GGML_SYCL_DIAG_HPP

#include "common.hpp"

void ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_DIAG_HPP
