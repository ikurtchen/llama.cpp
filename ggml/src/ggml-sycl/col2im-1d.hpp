#ifndef GGML_SYCL_COL2IM_1D_HPP
#define GGML_SYCL_COL2IM_1D_HPP

#include "common.hpp"

void ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_col2im_1d(const ggml_tensor * op);

#endif // GGML_SYCL_COL2IM_1D_HPP
