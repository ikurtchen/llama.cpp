#ifndef GGML_SYCL_CROSS_ENTROPY_LOSS_HPP
#define GGML_SYCL_CROSS_ENTROPY_LOSS_HPP

#include "common.hpp"

void ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_cross_entropy_loss(const ggml_tensor * op);
bool ggml_sycl_supports_cross_entropy_loss_back(const ggml_tensor * op);

#endif // GGML_SYCL_CROSS_ENTROPY_LOSS_HPP
