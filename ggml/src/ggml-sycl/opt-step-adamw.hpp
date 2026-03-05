#ifndef GGML_SYCL_OPT_STEP_ADAMW_HPP
#define GGML_SYCL_OPT_STEP_ADAMW_HPP

#include "common.hpp"

void ggml_sycl_op_opt_step_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_OPT_STEP_ADAMW_HPP
