# CUDA Kernel Categorization for llama.cpp SYCL Backend (Excluding MatMul)

Based on `kernel_spec.json` (106 CUDA kernels, 8 mul_mat + 7 flash attention kernels excluded).

## Table 1: Kernel Categories

| Category | Kernels | Count |
|----------|---------|-------|
| Dequantization | dequantize_block, dequantize_block_q8_0_f16, dequantize_block_q4_0, dequantize_block_q4_1, dequantize_block_q2_K, dequantize_block_q3_K, dequantize_block_q4_K, dequantize_block_q5_K, dequantize_block_q6_K, dequantize_block_iq2_xxs, dequantize_block_iq2_xs, dequantize_block_iq2_s, dequantize_block_iq3_xxs, dequantize_block_iq3_s, dequantize_block_iq1_s, dequantize_block_iq1_m, dequantize_block_iq4_nl, dequantize_block_iq4_xs, dequantize_block_mxfp4 | 19 |
| Tensor Manipulation | arange_f32, argmax_f32, init_indices, init_offsets, k_argsort_f32_i32, concat_f32_dim0, concat_f32_dim1, concat_f32_dim2, concat_f32_non_cont, diag_kernel, diag_mask_inf_f32, fill_kernel, pad_f32, pad_reflect_1d_kernel_f32, roll_f32_cuda, tri_kernel, upscale_f32, upscale_f32_bilinear, upscale_f32_bilinear_antialias, upscale_f32_bicubic | 20 |
| Unary / Activation | softcap_f32, unary_op_kernel, unary_gated_op_kernel, swiglu_oai_kernel, xielu_kernel, silu_back_kernel, leaky_relu_kernel | 7 |
| Binary / Element-wise Arithmetic | acc_f32, add_id_kernel, k_bin_bcast, k_bin_bcast_unravel, k_repeat_back, op_clamp_kernel, scale_f32 | 7 |
| Data Copy / Conversion | convert_unary, cpy_scalar, cpy_scalar_transpose, cpy_f32_q, cpy_q_f32, cpy_scalar_contiguous | 6 |
| Convolution | conv2d_kernel, conv2d_dw_kernel, conv2d_transpose_kernel, conv_transpose_1d_kernel, im2col_kernel, im2col_3d_kernel | 6 |
| Row Operations | count_equal, k_get_rows, k_get_rows_float, k_get_rows_back_float, k_set_rows_quant, k_set_rows | 6 |
| Loss & Training | cross_entropy_loss_f32, cross_entropy_loss_back_f32, cumsum_cub_kernel, cumsum_kernel, opt_step_adamw_f32, opt_step_sgd_f32 | 6 |
| Normalization | norm_f32, group_norm_f32, rms_norm_f32, rms_norm_back_f32, l2_norm_f32 | 5 |
| Positional Encoding (RoPE) | rope_norm, rope_neox, rope_multi, rope_vision | 4 |
| SSM (State Space Models) | ssm_conv_f32, ssm_conv_long_token_f32, ssm_scan_f32, ssm_scan_f32_group | 4 |
| Quantization | quantize_q8_1, quantize_mmq_mxfp4, quantize_mmq_q8_1 | 3 |
| Softmax | soft_max_f32, soft_max_back_f32, soft_max_f32_parallelize_cols | 3 |
| Recurrent / Linear Attention | gated_linear_attn_f32, rwkv_wkv_f32, rwkv_wkv7_f32 | 3 |
| Reduction | divide_by_count, reduce_rows_f32 | 2 |
| Linear Algebra | get_batch_pointers, solve_tri_f32_fast | 2 |
| Pooling | pool2d_nchw_kernel | 1 |
| MoE Routing | topk_moe_cuda | 1 |
| Embeddings | timestep_embedding_f32 | 1 |
| **Total** | | **106** |

## Table 2: Kernel Complexity Levels

| Complexity | Kernels | Count |
|------------|---------|-------|
| simple | acc_f32, add_id_kernel, arange_f32, argmax_f32, init_indices, init_offsets, op_clamp_kernel, concat_f32_dim0, concat_f32_dim1, concat_f32_dim2, dequantize_block_q8_0_f16, dequantize_block_q4_0, dequantize_block_q4_1, dequantize_block_q2_K, dequantize_block_q3_K, dequantize_block_q4_K, dequantize_block_q5_K, dequantize_block_q6_K, dequantize_block_iq4_nl, dequantize_block_iq4_xs, convert_unary, cpy_scalar_contiguous, cumsum_kernel, diag_kernel, diag_mask_inf_f32, fill_kernel, k_get_rows_float, divide_by_count, opt_step_sgd_f32, pad_f32, pad_reflect_1d_kernel_f32, roll_f32_cuda, scale_f32, k_set_rows, softcap_f32, get_batch_pointers, tri_kernel, timestep_embedding_f32, unary_op_kernel, unary_gated_op_kernel, swiglu_oai_kernel, xielu_kernel, silu_back_kernel, leaky_relu_kernel, upscale_f32 | 45 |
| medium | k_argsort_f32_i32, k_bin_bcast, k_bin_bcast_unravel, k_repeat_back, concat_f32_non_cont, conv2d_transpose_kernel, conv_transpose_1d_kernel, dequantize_block, dequantize_block_iq2_xxs, dequantize_block_iq2_xs, dequantize_block_iq2_s, dequantize_block_iq3_xxs, dequantize_block_iq3_s, dequantize_block_iq1_s, dequantize_block_iq1_m, dequantize_block_mxfp4, count_equal, cpy_scalar, cpy_scalar_transpose, cpy_f32_q, cpy_q_f32, cross_entropy_loss_f32, cross_entropy_loss_back_f32, k_get_rows, k_get_rows_back_float, im2col_kernel, im2col_3d_kernel, norm_f32, group_norm_f32, rms_norm_f32, rms_norm_back_f32, l2_norm_f32, opt_step_adamw_f32, pool2d_nchw_kernel, quantize_q8_1, quantize_mmq_mxfp4, quantize_mmq_q8_1, reduce_rows_f32, rope_norm, rope_neox, rope_multi, rope_vision, k_set_rows_quant, soft_max_f32, soft_max_back_f32, ssm_conv_f32, ssm_conv_long_token_f32, upscale_f32_bilinear, upscale_f32_bilinear_antialias, upscale_f32_bicubic | 50 |
| complex | conv2d_kernel, conv2d_dw_kernel, cumsum_cub_kernel, gated_linear_attn_f32, soft_max_f32_parallelize_cols, solve_tri_f32_fast, ssm_scan_f32, ssm_scan_f32_group, topk_moe_cuda, rwkv_wkv_f32, rwkv_wkv7_f32 | 11 |
| **Total** | | **106** |
