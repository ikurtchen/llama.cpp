#ifndef GGML_SYCL_TSEMBD_HPP
#define GGML_SYCL_TSEMBD_HPP

#include "common.hpp"

void ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_TSEMBD_HPP
