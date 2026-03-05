#ifndef GGML_SYCL_SUMROWS_HPP
#define GGML_SYCL_SUMROWS_HPP

#include "common.hpp"

void ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// Internal helper used by sum.cpp — reduces nrows rows of ncols elements each.
// When norm==true the result is divided by ncols (for MEAN).
void reduce_rows_f32_sycl(const float * x, float * dst, int ncols, int nrows,
                          bool norm, sycl::queue & stream);

#endif // GGML_SYCL_SUMROWS_HPP
