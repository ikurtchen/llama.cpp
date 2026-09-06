#ifndef GGML_SYCL_SOLVE_TRI_HPP
#define GGML_SYCL_SOLVE_TRI_HPP

#include "common.hpp"

void ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_solve_tri(const ggml_tensor * op);

#endif // GGML_SYCL_SOLVE_TRI_HPP
