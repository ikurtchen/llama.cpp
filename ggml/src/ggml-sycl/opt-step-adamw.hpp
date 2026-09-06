#ifndef GGML_SYCL_OPT_STEP_ADAMW_HPP
#define GGML_SYCL_OPT_STEP_ADAMW_HPP

#include "common.hpp"

void ggml_sycl_opt_step_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_opt_step_adamw(const ggml_tensor * op);

#endif // GGML_SYCL_OPT_STEP_ADAMW_HPP
