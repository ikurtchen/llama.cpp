#pragma once

#include "common.hpp"

void ggml_sycl_op_softcap(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst);