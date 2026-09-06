#ifndef GGML_SYCL_ARGSORT_HPP
#define GGML_SYCL_ARGSORT_HPP

#include "common.hpp"

void ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_argsort(const ggml_tensor * op);

// Exposed for reuse by top-k (sort descending, then take the first k columns per row).
void ggml_sycl_argsort_bitonic_raw(sycl::queue & q, const float * x, int * dst, int ncols, int ncols_pad,
                                    int nrows, bool ascending);
int ggml_sycl_next_power_of_2(int x);

#endif // GGML_SYCL_ARGSORT_HPP
