#pragma once

#include "common.hpp"

bool ggml_sycl_lightning_indexer_supported(int device, const ggml_tensor * dst);
void ggml_sycl_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst);