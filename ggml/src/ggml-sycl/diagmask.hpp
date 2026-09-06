#ifndef GGML_SYCL_DIAGMASK_HPP
#define GGML_SYCL_DIAGMASK_HPP

#include "common.hpp"

void ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_diag_mask_inf(const ggml_tensor * op);

#endif // GGML_SYCL_DIAGMASK_HPP
