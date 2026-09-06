#ifndef GGML_SYCL_LIGHTNING_INDEXER_HPP
#define GGML_SYCL_LIGHTNING_INDEXER_HPP

#include "common.hpp"

void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_supports_lightning_indexer(const ggml_tensor * op);

#endif // GGML_SYCL_LIGHTNING_INDEXER_HPP
