#ifndef GGML_SYCL_SOFTCAP_HPP
#define GGML_SYCL_SOFTCAP_HPP

#include "common.hpp"

void ggml_sycl_op_softcap(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * src);
bool ggml_sycl_can_fuse_softcap(const ggml_cgraph * cgraph, int node_idx);

#endif // GGML_SYCL_SOFTCAP_HPP
