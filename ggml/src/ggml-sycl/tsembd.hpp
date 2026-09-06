#ifndef GGML_SYCL_TSEMBD_HPP
#define GGML_SYCL_TSEMBD_HPP

#include "common.hpp"

void ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_timestep_embedding(const ggml_tensor * op);

#endif // GGML_SYCL_TSEMBD_HPP
