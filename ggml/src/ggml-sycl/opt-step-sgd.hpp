#ifndef GGML_SYCL_OPT_STEP_SGD_HPP
#define GGML_SYCL_OPT_STEP_SGD_HPP

#include "common.hpp"

void ggml_sycl_opt_step_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_opt_step_sgd(const ggml_tensor * op);

#endif // GGML_SYCL_OPT_STEP_SGD_HPP
