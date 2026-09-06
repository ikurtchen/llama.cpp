#ifndef GGML_SYCL_SNAKE_HPP
#define GGML_SYCL_SNAKE_HPP

#include "common.hpp"

void ggml_sycl_op_snake_fused(ggml_backend_sycl_context & ctx,
                              const ggml_tensor * x,
                              const ggml_tensor * a,
                              const ggml_tensor * inv_b,
                              ggml_tensor * dst);

bool ggml_sycl_can_fuse_snake(const ggml_cgraph * cgraph, int node_idx);

#endif // GGML_SYCL_SNAKE_HPP
