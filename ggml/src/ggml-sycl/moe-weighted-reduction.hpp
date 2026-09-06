#pragma once

#include "common.hpp"

int ggml_sycl_try_moe_weighted_reduction(
        ggml_backend_sycl_context & ctx,
        ggml_cgraph * cgraph,
        int node_idx);
