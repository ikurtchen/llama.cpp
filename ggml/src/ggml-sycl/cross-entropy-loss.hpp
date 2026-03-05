#ifndef GGML_SYCL_CROSS_ENTROPY_LOSS_HPP
#define GGML_SYCL_CROSS_ENTROPY_LOSS_HPP

#include "common.hpp"

void ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_CROSS_ENTROPY_LOSS_HPP
