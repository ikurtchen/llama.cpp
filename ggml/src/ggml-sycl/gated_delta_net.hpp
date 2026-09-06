#ifndef GGML_SYCL_GATED_DELTA_NET_HPP
#define GGML_SYCL_GATED_DELTA_NET_HPP

#include "common.hpp"

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_gated_delta_net(const ggml_tensor * op);

#endif // GGML_SYCL_GATED_DELTA_NET_HPP
