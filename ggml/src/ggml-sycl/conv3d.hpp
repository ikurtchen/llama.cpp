#ifndef GGML_SYCL_CONV3D_HPP
#define GGML_SYCL_CONV3D_HPP

#include "common.hpp"

void ggml_sycl_op_conv3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_conv3d(const ggml_tensor * op);

#endif // GGML_SYCL_CONV3D_HPP
