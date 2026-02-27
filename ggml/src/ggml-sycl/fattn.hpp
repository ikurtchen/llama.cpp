//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_FATTN_HPP
#define GGML_SYCL_FATTN_HPP

#include "ggml-sycl.h"
#include "ggml-backend-impl.h"
#include "ggml-sycl/common.hpp"

void ggml_sycl_op_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_HPP
