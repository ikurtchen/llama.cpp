#!/usr/bin/env python3
"""Generate all CUDA kernel detail files and index.json for the sycl-agent inventory."""

import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.environ.get("ROOT_DIR", os.getcwd())
STATE = os.path.join(ROOT, ".sycl", "state")
KERNELS_DIR = os.path.join(STATE, "kernels")
CUDA_SRC = os.path.join(ROOT, "ggml", "src", "ggml-cuda")

os.makedirs(KERNELS_DIR, exist_ok=True)

now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

DEFAULT_REF = {
    "kind": "cpu",
    "notes": "llama.cpp CPU (ggml) backend produces the reference output"
}

# ============================================================
# Kernel definitions: id, name, source, algorithm, risk, cuda_constructs, io_spec, notes
# After grouping template instantiations
# ============================================================

KERNELS = [
    # --- Element-wise / simple kernels (LOW risk) ---
    {
        "id": "acc-f32",
        "name": "acc_f32",
        "source": "ggml/src/ggml-cuda/acc.cu:3",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Per-element accumulation: dst[i] = x[i] + y[i] * s1, with support for broadcast dimensions across up to 4 axes (ne10..ne13).",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const float* [ne]", "y": "const float* [ne10*ne11*ne12*ne13 broadcast]"},
            "outputs": {"dst": "float* [ne]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Simple element-wise with broadcast. Straightforward SYCL mapping."
    },
    {
        "id": "add-id",
        "name": "add_id_kernel",
        "source": "ggml/src/ggml-cuda/add-id.cu:3",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Scatter-add by ID: dst[row] += src[id_row] for each row, indexed by an ID array. Used for embedding row updates.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {
                "x": "const float* [num_rows, ncols]",
                "ids": "const int* [num_rows_to_add]"
            },
            "outputs": {"dst": "float* [num_rows, ncols]"},
            "dtypes": "f32, i32"
        },
        "reference": DEFAULT_REF,
        "notes": "Scatter-add pattern, straightforward."
    },
    {
        "id": "arange-f32",
        "name": "arange_f32",
        "source": "ggml/src/ggml-cuda/arange.cu:3",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Fill contiguous output with arithmetic progression: dst[i] = start + i * step.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {},
            "outputs": {"dst": "float* [ne0]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Trivial fill kernel."
    },
    {
        "id": "clamp",
        "name": "op_clamp_kernel",
        "source": "ggml/src/ggml-cuda/clamp.cu:8",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Clamp each element: dst[i] = min(max(x[i], min_val), max_val).",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T* [k]", "min": "T", "max": "T"},
            "outputs": {"dst": "T* [k]"},
            "dtypes": "T = f32 or f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Simple bounded clamp."
    },
    {
        "id": "diag",
        "name": "diag_kernel",
        "source": "ggml/src/ggml-cuda/diag.cu:6",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Extract or set diagonal elements of a matrix. Each thread handles one diagonal element, mapping 1D index to 2D positions.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src0": "const T* [nrows, ncols]"},
            "outputs": {"dst": "T* [ne0]"},
            "dtypes": "T = f32 or f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Simple diagonal extraction."
    },
    {
        "id": "diagmask",
        "name": "diag_mask_inf_f32",
        "source": "ggml/src/ggml-cuda/diagmask.cu:3",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Apply causal diagonal mask for attention: sets future positions to -inf (past-context masking). Used in autoregressive attention.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const float* [ncols, nrows_per_channel]"},
            "outputs": {"dst": "float* [same shape]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Simple causal masking."
    },
    {
        "id": "fill",
        "name": "fill_kernel",
        "source": "ggml/src/ggml-cuda/fill.cu:7",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Fill array with a constant scalar value: dst[i] = value for all i.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"value": "T"},
            "outputs": {"dst": "T* [k]"},
            "dtypes": "T = f32 or f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Trivial fill."
    },
    {
        "id": "scale-f32",
        "name": "scale_f32",
        "source": "ggml/src/ggml-cuda/scale.cu:5",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Scale and bias: dst[i] = x[i] * scale + bias.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const float* [nelements]"},
            "outputs": {"dst": "float* [nelements]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Standard affine transform."
    },
    {
        "id": "softcap-f32",
        "name": "softcap_f32",
        "source": "ggml/src/ggml-cuda/softcap.cu:3",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Apply softcapping: dst[i] = softcap * tanh(x[i] / softcap) * scale. Prevents logits from growing unboundedly.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const float* [k]", "scale": "float", "softcap": "float"},
            "outputs": {"dst": "float* [k]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Simple per-element transcendentals."
    },
    {
        "id": "snake",
        "name": "snake_kernel",
        "source": "ggml/src/ggml-cuda/snake.cu:9",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Snake activation: snake(x) = x + (1/theta) * sin^2(theta*x). Also computes the derivative for backprop. Templated on forward/backward mode.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T* [k]", "theta": "float"},
            "outputs": {"dst": "T* [k]"},
            "dtypes": "T = f32 or f16 or bf16"
        },
        "reference": DEFAULT_REF,
        "notes": "Activation function, straightforward."
    },
    {
        "id": "roll-f32",
        "name": "roll_f32_cuda",
        "source": "ggml/src/ggml-cuda/roll.cu:14",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Circular shift ('roll') of elements along up to 3 dimensions. Each thread computes source index via wrap-around modulo.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src": "const float* [ne0, ne1, ne2, ne3]", "shifts": "int[3]"},
            "outputs": {"dst": "float* [same shape]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Wrap-around index arithmetic."
    },
    {
        "id": "tsembd",
        "name": "timestep_embedding_f32",
        "source": "ggml/src/ggml-cuda/tsembd.cu:3",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Compute sinusoidal timestep embedding for diffusion models: uses sin/cos at geometrically-spaced frequencies per max_period.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"timesteps": "const float* [ne0]", "max_period": "int", "dim": "int"},
            "outputs": {"dst": "float* [ne0 * dim]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Sinusoidal embedding, straightforward."
    },
    {
        "id": "fwht",
        "name": "fwht_cuda",
        "source": "ggml/src/ggml-cuda/fwht.cu:6",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Fast Walsh-Hadamard Transform. Recursive butterfly network: each thread computes its own row using shuffle_xor to exchange data across warps.",
        "cuda_constructs": ["warp-shuffle"],
        "io_spec": {
            "inputs": {"src": "const float* [n_rows, ne]", "scale": "float"},
            "outputs": {"dst": "float* [same shape]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Uses warp shuffle for butterfly exchange. Map to sycl::group_broadcast/shuffle."
    },

    # --- Dequantize / type-conversion kernels (LOW-MEDIUM) ---
    {
        "id": "dequantize-block",
        "name": "dequantize_block (generic)",
        "source": "ggml/src/ggml-cuda/convert.cu:9",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Generic dequantize block: converts a quantized block (q4_0/q4_1/q5_0/q5_1/q8_0 format) to float. Uses __shfl_sync-based warp-level dequantize for certain formats.",
        "cuda_constructs": ["warp-shuffle", "__shared__"],
        "io_spec": {
            "inputs": {"vx": "const void* [quantized blocks]", "k": "int64_t"},
            "outputs": {"y": "dst_t* [k]"},
            "dtypes": "dst_t = f32 or f16; quant formats: q4_0, q4_1, q5_0, q5_1, q8_0"
        },
        "reference": DEFAULT_REF,
        "variants": [
            "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
            "to f32", "to f16"
        ],
        "notes": "Template-based dequant. Has warp-shuffle data exchange for QK_4_0/QK_4_1 variants."
    },
    {
        "id": "dequantize-block-q8-0-f16",
        "name": "dequantize_block_q8_0_f16",
        "source": "ggml/src/ggml-cuda/convert.cu:44",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Specialized dequantize q8_0 -> f16 using warp-cooperative reduction. Each warp dequantizes one block with __shfl_sync exchanges for the scale/d value.",
        "cuda_constructs": ["warp-shuffle", "__shared__"],
        "io_spec": {
            "inputs": {"vx": "const void* [q8_0 blocks]", "k": "int64_t"},
            "outputs": {"y": "half* [k]"},
            "dtypes": "q8_0 -> f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Warp-level cooperative dequant. Needs sycl::group_broadcast equivalent."
    },
    {
        "id": "dequantize-block-per-type",
        "name": "dequantize_block_* (per-type)",
        "source": "ggml/src/ggml-cuda/convert.cu (multi)",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Per-quantization-type dequantize kernels: q2_K, q3_K, q4_K, q5_K, q6_K, IQ variants (iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs), mxfp4, nvfp4. Each converts packed-bit representations to float.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"vx": "const void* [quantized blocks]", "nb": "int64_t (for some)"},
            "outputs": {"yy": "float* [dequantized]"},
            "dtypes": "various quant types (q2_K..q6_K, iq*, mxfp4, nvfp4) -> f32"
        },
        "reference": DEFAULT_REF,
        "variants": [
            "q4_0", "q4_1", "q2_K", "q3_K", "q4_K", "q5_K", "q6_K",
            "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s",
            "iq1_s", "iq1_m", "iq4_nl", "iq4_xs",
            "mxfp4", "nvfp4"
        ],
        "notes": "~19 per-type kernels. Most are straightforward unpacking loops, but K-quant formats involve multi-level scale hierarchies."
    },
    {
        "id": "convert-unary",
        "name": "convert_unary",
        "source": "ggml/src/ggml-cuda/convert.cu:417",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Unary type conversion: cast elements from src_t to dst_t. Used for float <-> f16 conversion.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const src_t* [k]"},
            "outputs": {"y": "float* [k]"},
            "dtypes": "src_t -> f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Simple type cast."
    },

    # --- Copy kernels ---
    {
        "id": "cpy-scalar",
        "name": "cpy_scalar",
        "source": "ggml/src/ggml-cuda/cpy.cu:15",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Strided copy with up to 4-dimensional source/destination. Each thread copies one element, computing offsets from stride parameters.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"cx": "const char* [ne00*ne01*ne02*ne03]"},
            "outputs": {"cdst": "char* [same]"},
            "dtypes": "byte-copy (generic)"
        },
        "reference": DEFAULT_REF,
        "notes": "Generic strided copy."
    },
    {
        "id": "cpy-scalar-transpose",
        "name": "cpy_scalar_transpose",
        "source": "ggml/src/ggml-cuda/cpy.cu:45",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Strided copy with transpose of the first two dimensions. Uses __shared__ memory tile for coalesced reads.",
        "cuda_constructs": ["__shared__"],
        "io_spec": {
            "inputs": {"cx": "const char* [ne00*ne01]"}, 
            "outputs": {"cdst": "char* [ne01*ne00]"},
            "dtypes": "byte-copy (generic)"
        },
        "reference": DEFAULT_REF,
        "notes": "Uses shared memory tiling for coalescing."
    },
    {
        "id": "cpy-scalar-contiguous",
        "name": "cpy_scalar_contiguous",
        "source": "ggml/src/ggml-cuda/cpy.cu:180",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Contiguous memcpy (byte-level). Simple bulk copy with no strides.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"cx": "const char* [ne]"},
            "outputs": {"cdst": "char* [ne]"},
            "dtypes": "byte-copy"
        },
        "reference": DEFAULT_REF,
        "notes": "Equivalent to memcpy."
    },
    {
        "id": "cpy-f32-q",
        "name": "cpy_f32_q",
        "source": "ggml/src/ggml-cuda/cpy.cu:126",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Quantize: float -> quantized block. Each thread quantizes one block of 32 floats into a packed quantized format.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"cx": "const char* (f32) [ne]"},
            "outputs": {"cdst": "char* (quantized) [ne / block_size]"},
            "dtypes": "f32 -> q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl"
        },
        "reference": DEFAULT_REF,
        "notes": "Quantization path, block-granular."
    },
    {
        "id": "cpy-q-f32",
        "name": "cpy_q_f32",
        "source": "ggml/src/ggml-cuda/cpy.cu:153",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Dequantize: quantized block -> float. Each thread dequantizes one block.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"cx": "const char* (quantized) [nb_blocks]"},
            "outputs": {"cdst": "char* (f32) [nb_blocks * block_size]"},
            "dtypes": "q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl -> f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Dequantization path, block-granular."
    },

    # --- Unary / activation kernels ---
    {
        "id": "unary-op",
        "name": "unary_op_kernel",
        "source": "ggml/src/ggml-cuda/unary.cu:118",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Apply a single unary function element-wise: abs, neg, step, gelu, silu, tanh, relu, sigmoid, exp, sqr, sqrt, sin, cos, log, softplus, elu, floor, ceil, round, trunc. Template dispatch selects the op.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T* [k]"},
            "outputs": {"dst": "T* [k]"},
            "dtypes": "T = f32, f16, bf16"
        },
        "reference": DEFAULT_REF,
        "variants": ["20+ ops via template enum"],
        "notes": "Generic element-wise unary. Large number of op variants."
    },
    {
        "id": "unary-gated-op",
        "name": "unary_gated_op_kernel",
        "source": "ggml/src/ggml-cuda/unary.cu:263",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Gated activation: dst[i] = op(x[i]) * g[i] where op is silu/gelu etc. Used for GLU variants.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T* [k]", "g": "const T* [k]"},
            "outputs": {"dst": "T* [k]"},
            "dtypes": "T = f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Element-wise gated activation."
    },
    {
        "id": "swiglu-oai",
        "name": "swiglu_oai_kernel",
        "source": "ggml/src/ggml-cuda/unary.cu:363",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "OpenAI-style SwiGLU: dst = (sigmoid(g) * x)_left_half || (sigmoid(g)_right_half * x_right_half). Splits tensor at n/2 boundary.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T*", "g": "const T*"},
            "outputs": {"dst": "T*"},
            "dtypes": "T = f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Swish-GLU variant."
    },
    {
        "id": "xielu",
        "name": "xielu_kernel",
        "source": "ggml/src/ggml-cuda/unary.cu:433",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "XIELU activation: piecewise linear+exponential with separate alpha_n/alpha_p for negative/positive slopes, beta for exponential tail, eps for regularization.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T* [k]", "alpha_n": "float", "alpha_p": "float", "beta": "float", "eps": "float"},
            "outputs": {"dst": "T* [k]"},
            "dtypes": "T = f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Novel activation, simple element-wise."
    },
    {
        "id": "silu-back",
        "name": "silu_back_kernel",
        "source": "ggml/src/ggml-cuda/unary.cu:491",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "SiLU backward pass: dst[i] = grad[i] * d_silu(x[i]) where d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))).",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"grad": "const T* [k]", "xf": "const T* [k] (forward activations)"},
            "outputs": {"dst": "T* [k]"},
            "dtypes": "T = f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Backward pass, straightforward."
    },
    {
        "id": "leaky-relu",
        "name": "leaky_relu_kernel",
        "source": "ggml/src/ggml-cuda/unary.cu:537",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Leaky ReLU: dst[i] = x[i] > 0 ? x[i] : x[i] * negative_slope.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T* [k]", "negative_slope": "float"},
            "outputs": {"dst": "T* [k]"},
            "dtypes": "T = f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Simple conditional activation."
    },

    # --- Binary / broadcast kernels ---
    {
        "id": "bin-bcast",
        "name": "k_bin_bcast",
        "source": "ggml/src/ggml-cuda/binbcast.cu:34",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Element-wise binary op with broadcasting (add/sub/mul/div). Uses 1D indexing with broadcast index unwrapping. Template param selects op.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src0": "const src0_t* [broadcast shape]", "src1": "const src1_t* [full shape]"},
            "outputs": {"dst": "dst_t* [full shape]"},
            "dtypes": "f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Standard element-wise binary with broadcast."
    },
    {
        "id": "bin-bcast-unravel",
        "name": "k_bin_bcast_unravel",
        "source": "ggml/src/ggml-cuda/binbcast.cu:106",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Binary op with broadcasting, alternate path that unravels the 1D thread index into multi-dimensional coordinates for partial broadcasting.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src0": "const src0_t*", "src1": "const src1_t*"},
            "outputs": {"dst": "dst_t*"},
            "dtypes": "f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Alternate broadcast path for non-contiguous cases."
    },
    {
        "id": "repeat-back",
        "name": "k_repeat_back",
        "source": "ggml/src/ggml-cuda/binbcast.cu:359",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Backward of repeat: reduce repeated dims back to original shape via summation.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src0": "const T* [repeated shape]"},
            "outputs": {"dst": "T* [original shape]"},
            "dtypes": "f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Reverse-broadcast reduction."
    },
    {
        "id": "tri",
        "name": "tri_kernel",
        "source": "ggml/src/ggml-cuda/tri.cu:7",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Generate upper/lower triangular matrix from source. Templated on upper/lower and diagonal offset.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T* [nrows, ncols]"},
            "outputs": {"dst": "T* [nrows, ncols]"},
            "dtypes": "T = f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Triangular mask generation."
    },
    {
        "id": "set-rows",
        "name": "k_set_rows",
        "source": "ggml/src/ggml-cuda/set-rows.cu:114",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Scatter copy rows from src0 into dst at positions specified by src1 indices.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src0_ptr": "const src_t* [nrows_src, ncols]", "src1_ptr": "const int* [nrows]"},
            "outputs": {"dst_ptr": "src_t* [nrows_dst, ncols]"},
            "dtypes": "f32, f16, i32"
        },
        "reference": DEFAULT_REF,
        "notes": "Gather-scatter copy."
    },
    {
        "id": "set-rows-quant",
        "name": "k_set_rows_quant",
        "source": "ggml/src/ggml-cuda/set-rows.cu:8",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Quantize float rows and scatter into quantized dst at indexed positions. Combines quantization with scatter-copy.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src0": "const float* [nrows_src, ncols_f32]", "ids": "const int*"},
            "outputs": {"dst": "block_type* [nrows_dst, ncols/blocksize]"},
            "dtypes": "f32 -> quantized"
        },
        "reference": DEFAULT_REF,
        "notes": "Quantize + scatter."
    },

    # --- Reduction kernels (MEDIUM) ---
    {
        "id": "norm-f32",
        "name": "norm_f32",
        "source": "ggml/src/ggml-cuda/norm.cu:5",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Layer normalization: computes mean and variance per row via block-level warp reduction (warp_reduce_sum), then normalizes each element. Uses shared memory for partial sums.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_sum", "warp_reduce_max"],
        "io_spec": {
            "inputs": {"x": "const float* [nrows, ncols]", "eps": "float"},
            "outputs": {"dst": "float* [same shape]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Block-level reduction with warp primitives. Map to sycl::reduce_over_group."
    },
    {
        "id": "group-norm-f32",
        "name": "group_norm_f32",
        "source": "ggml/src/ggml-cuda/norm.cu:42",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Group normalization: divides channels into groups, normalizes within each group. Block-level reduction per group.",
        "cuda_constructs": ["warp-shuffle", "__shared__"],
        "io_spec": {
            "inputs": {"x": "const float* [ne_elements]", "group_size": "int", "eps": "float"},
            "outputs": {"dst": "float* [same shape]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Similar to layer norm but grouped."
    },
    {
        "id": "rms-norm-f32",
        "name": "rms_norm_f32",
        "source": "ggml/src/ggml-cuda/norm.cu:77",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "RMS normalization: computes RMS (root-mean-square) per row via warp_reduce_sum, then scales each element: dst[i] = x[i] / sqrt(rms + eps). Optionally with fused gating.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {"x": "const float* [nrows, ncols]", "eps": "float", "w": "const float* (optional, for gating)"},
            "outputs": {"dst": "float* [same shape]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Common in LLMs (LLaMA, etc.). Fused gating variant."
    },
    {
        "id": "rms-norm-back-f32",
        "name": "rms_norm_back_f32",
        "source": "ggml/src/ggml-cuda/norm.cu:158",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Backward pass for RMS norm: computes gradient w.r.t input given gradient w.r.t output and forward activations. Uses warp_reduce_sum for inner products xx, xg.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {"grad": "const float* [nrows, ncols]", "xf": "const float* [nrows, ncols] (forward output)", "eps": "float"},
            "outputs": {"dst": "float* [nrows, ncols]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Backward of RMS norm."
    },
    {
        "id": "l2-norm-f32",
        "name": "l2_norm_f32",
        "source": "ggml/src/ggml-cuda/norm.cu:245",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "L2 norm per row: dst_row[i] = x_row[i] / max(sqrt(sum(x_row^2)), eps). Uses warp_reduce_sum.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {"x": "const float* [nrows, ncols]", "eps": "float"},
            "outputs": {"dst": "float* [same shape]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Normalization by L2 vector norm."
    },
    {
        "id": "reduce-rows-f32",
        "name": "reduce_rows_f32",
        "source": "ggml/src/ggml-cuda/reduce_rows.cuh:5",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Row-wise sum reduction: dst[i] = sum over j of x[i, j]. Uses warp_reduce_sum within each row.",
        "cuda_constructs": ["warp-shuffle", "__shared__"],
        "io_spec": {
            "inputs": {"x_ptr": "const float* [nrows, ncols]"},
            "outputs": {"dst_ptr": "float* [nrows]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Standard row reduction."
    },
    {
        "id": "sum",
        "name": "ggml_cuda_sum (via CUB)",
        "source": "ggml/src/ggml-cuda/sum.cu",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Sum reduction across rows via CUB device-level reduce. Uses cuBLAS if available for certain cases.",
        "cuda_constructs": ["cub", "cublas (optional)"],
        "io_spec": {
            "inputs": {"src": "float* [nrows, ncols]"},
            "outputs": {"dst": "float* [nrows]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Relies on CUB. SYCL has oneDPL equivalent."
    },
    {
        "id": "sumrows",
        "name": "sumrows (via CUB)",
        "source": "ggml/src/ggml-cuda/sumrows.cu",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Sum reduction within each row (keeping row dimension). Uses CUB device-level segmented reduce.",
        "cuda_constructs": ["cub"],
        "io_spec": {
            "inputs": {"src": "const float* [nrows, ncols]"},
            "outputs": {"dst": "float* [nrows, ncols] (per-row sum broadcast)"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Depends on CUB segmented reduce."
    },
    {
        "id": "mean-divide",
        "name": "divide_by_count",
        "source": "ggml/src/ggml-cuda/mean.cu:9",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Final step of mean: divide accumulated sum by count. Single-element kernel launched with 1 thread.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"result": "T* [1]", "count": "size_t"},
            "outputs": {"result": "T (in-place)"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Trivial single-element division."
    },

    # --- Softmax / attention helpers ---
    {
        "id": "soft-max-f32",
        "name": "soft_max_f32",
        "source": "ggml/src/ggml-cuda/softmax.cu:55",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Online numerically-stable softmax: uses warp_reduce_max for max finding, warp_reduce_sum for denominator. Supports optional mask (-inf masking), sinks, and logit-scale. Computes softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max)).",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_max", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {"x": "const float* [nrows, ncols]", "mask": "const float* (optional)", "sinks": "const float* (optional)", "scale": "float"},
            "outputs": {"dst": "float* [same shape]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Core attention primitive. Warp-level reductions. The online algorithm is numerically important."
    },
    {
        "id": "soft-max-f32-parallelize-cols",
        "name": "soft_max_f32_parallelize_cols",
        "source": "ggml/src/ggml-cuda/softmax.cu:302",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Parallel-columns softmax using cooperative groups grid-level sync. Multiple thread blocks collaborate across columns using cg::grid_group for synchronization. Each block computes a partial softmax, then grid-level reduction combines results.",
        "cuda_constructs": ["cooperative-groups", "warp-shuffle", "__shared__"],
        "io_spec": {
            "inputs": {"x": "const float*", "mask": "const float* (opt)", "sinks": "const float* (opt)", "params": "struct"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Cooperative-groups grid sync is the hardest CUDA construct to port. SYCL equivalent is experimental/device-specific."
    },
    {
        "id": "soft-max-back-f32",
        "name": "soft_max_back_f32",
        "source": "ggml/src/ggml-cuda/softmax.cu:250",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Softmax backward pass: dst = grad * dst_fwd - dst_fwd * sum(grad * dst_fwd). Uses warp_reduce_sum for inner product.",
        "cuda_constructs": ["warp-shuffle", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {"grad": "const float* [nrows, ncols]", "dstf": "const float* [nrows, ncols] (forward output)", "scale": "float"},
            "outputs": {"dst": "float* [same shape]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Backward of online softmax."
    },

    # --- Attention kernels (HIGH complexity) ---
    {
        "id": "flash-attn-ext-vec",
        "name": "flash_attn_ext_vec",
        "source": "ggml/src/ggml-cuda/fattn-vec.cuh:21",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Flash attention (vector variant): each thread computes attention for one query vector against all K/V. Uses shared memory for KQ tile, warp reductions for max/sum (online softmax), and supports quantized K/V (dequantized on-the-fly). Supports stream-k splitting.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_max", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {
                "Q": "const Dkq_t* [ncols, DKQ]",
                "K": "const Dkv_t* [ne02, DKQ]",
                "V": "const Dkv_t* [ne02, DV]",
                "mask": "const float* (optional)"
            },
            "outputs": {"dst": "float* [ncols, DV]"},
            "dtypes": "Dkq_t = f16/bf16/q*, Dkv_t = f16/bf16/q*"
        },
        "reference": DEFAULT_REF,
        "variants": ["49 type-pair instantiations: (f16,bf16,q4_0,q4_1,q5_0,q5_1,q8_0) x (same set)"],
        "notes": "Template instances in fattn-vec-instance-*.cu. The 49 type combos are all the same kernel template with different KQ dequant and V dequant functions."
    },
    {
        "id": "flash-attn-tile",
        "name": "flash_attn_tile",
        "source": "ggml/src/ggml-cuda/fattn-tile.cuh:793",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Tiled flash attention: each block processes a tile of queries against a tile of K/V. Uses shared memory for K/V tiles, online softmax with warp-level reductions, and device-configurable tile sizes based on DKQ/DV dimensions. Supports fp16 and fp32 compute.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_max", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {
                "Q": "const T* [ncols, DKQ]",
                "K": "const T* [n_kv, DKQ]",
                "V": "const T* [n_kv, DV]",
                "mask": "const float* (optional)"
            },
            "outputs": {"dst": "float* [ncols, DV]"},
            "dtypes": "f16 or f32"
        },
        "reference": DEFAULT_REF,
        "variants": ["12 size instances: dkq={40,64,72,80,96,112,128,192,256,320,512,576}, dv per config"],
        "notes": "Template instances in fattn-tile-instance-*.cu. Each instance is the same kernel with different compile-time DKQ/DV parameters."
    },
    {
        "id": "flash-attn-ext-f16",
        "name": "flash_attn_ext_f16",
        "source": "ggml/src/ggml-cuda/fattn-mma-f16.cuh:1705",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Flash attention with Tensor Core / MMA acceleration. Uses warp-group matrix multiply-accumulate (wgmma/mma instructions) for Q*K^T computation. Supports Hopper wgmma, pre-Hopper mma.sync, AMD MFMA, and CDNA path. Complex tiling with multi-stage async copy pipelines, cooperative partial reduction.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "mma.sync", "wgmma", "cp-async", "warp_reduce_max", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {
                "Q": "const half* [ncols1 * DKQ]",
                "K": "const half* [n_kv * DKQ]",
                "V": "const half* [n_kv * DV]",
                "mask": "const float* (optional)"
            },
            "outputs": {"dst": "float* [ncols1, DV]"},
            "dtypes": "f16 / half"
        },
        "reference": DEFAULT_REF,
        "variants": ["21 combos of ncols1={1,2,4,8,16,32,64} x ncols2"],
        "notes": "Most complex attention kernel. Uses MMA/warp-group MMA. Hardest to port. Template instances in fattn-mma-f16-instance-*.cu."
    },
    {
        "id": "flash-attn-mask-to-KV-max",
        "name": "flash_attn_mask_to_KV_max",
        "source": "ggml/src/ggml-cuda/fattn-common.cuh:666",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Pre-process attention mask: computes per-KV maximum value, used by stream-k attention fixup.",
        "cuda_constructs": ["__shared__", "warp-shuffle"],
        "io_spec": {
            "inputs": {"mask": "const float* [ne_mask]"}, 
            "outputs": {"buf_iw": "int* [WARP_SIZE]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Mask preprocessing helper for stream-k attention."
    },
    {
        "id": "flash-attn-stream-k-fixup-uniform",
        "name": "flash_attn_stream_k_fixup_uniform",
        "source": "ggml/src/ggml-cuda/fattn-common.cuh:723",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Stream-K attention fixup (uniform split): combines partial results from multiple stream-k splits using online softmax merging. Reduces KV_max across all stream-k results.",
        "cuda_constructs": ["warp-shuffle", "__shared__"],
        "io_spec": {
            "inputs": {"meta": "float2* [partial results]", "KQ_max": "float*", "KQ_sum": "float*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Stream-K merge kernel."
    },
    {
        "id": "flash-attn-stream-k-fixup-general",
        "name": "flash_attn_stream_k_fixup_general",
        "source": "ggml/src/ggml-cuda/fattn-common.cuh:807",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "General stream-k attention fixup: handles non-uniform K/V split merging with more complex max/sum aggregation across partial attention outputs.",
        "cuda_constructs": ["warp-shuffle", "__shared__"],
        "io_spec": {
            "inputs": {"meta": "float2*", "partial_KQ_max": "float*", "partial_KQ_sum": "float*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "More general stream-k merge."
    },
    {
        "id": "flash-attn-combine-results",
        "name": "flash_attn_combine_results",
        "source": "ggml/src/ggml-cuda/fattn-common.cuh:916",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Combines attention results from multiple parallel blocks (e.g., from stream-k splits or parallel-columns). Uses shared memory for KQ_max/KQ_sum aggregation and weighted combination of partial output vectors.",
        "cuda_constructs": ["__shared__", "warp-shuffle"],
        "io_spec": {
            "inputs": {"meta": "float2* [nwarps * ncols * ne_combine]"}, 
            "outputs": {"dst": "float* [ncols * DV]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Result combination for multi-block attention."
    },

    # --- GEMM / Matrix Multiply kernels (HIGH complexity) ---
    {
        "id": "mul-mat-q",
        "name": "mul_mat_q",
        "source": "ggml/src/ggml-cuda/mmq.cuh:934",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Quantized matrix-matrix multiply (D = dequant(Q) * X). Uses configurable tile sizes per GPU architecture (Ampere/Blackwell/CDNA/RDNA/Pascal). Supports dp4a and MMA (Tensor Core) compute paths. Includes stream-k variant for load-balanced scheduling. Complex: extern shared memory, multi-level tiling, warp-level matrix operations.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "mma.sync", "dp4a", "warp_reduce_sum", "cp-async (stream-k)"],
        "io_spec": {
            "inputs": {
                "src0": "const char* (quantized, type-dependent) [M, K]",
                "src1": "const float* [K, N]",
                "ids": "const int* (optional expert IDs)"
            },
            "outputs": {"dst": "float* [M, N]"},
            "dtypes": "quantized input (21 quant types), f32 output"
        },
        "reference": DEFAULT_REF,
        "variants": ["21 quant types: q1_0, q2_K, q3_K, q4_0, q4_1, q4_K, q5_0, q5_1, q5_K, q6_K, q8_0, iq1_s, iq2_s, iq2_xs, iq2_xxs, iq3_s, iq3_xxs, iq4_nl, iq4_xs, mxfp4, nvfp4"],
        "notes": "Most critical performance kernel. Uses per-architecture config files (mmq-config-*.cuh). Template instances in mmq-instance-*.cu. Uses extern shared memory and MMA hardware."
    },
    {
        "id": "mul-mat-q-stream-k-fixup",
        "name": "mul_mat_q_stream_k_fixup",
        "source": "ggml/src/ggml-cuda/mmq.cuh:1221",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Fixup kernel for stream-k mul_mat_q: corrects the boundary rows where multiple stream-k tiles' partial results overlap. Scales row contributions from adjacent stream-k tiles.",
        "cuda_constructs": ["__shared__"],
        "io_spec": {
            "inputs": {"dst": "float* [M, N] (partial)", "ids_dst_shared": "int*"},
            "outputs": {"dst": "float* (in-place fixup)"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Stream-k boundary correction."
    },
    {
        "id": "mul-mat-f",
        "name": "mul_mat_f",
        "source": "ggml/src/ggml-cuda/mmf.cuh:50",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Matrix-matrix multiply for small ncols (D = X * Y where ncols_Y is 1-16). Each block processes rows_per_block rows of output. Uses shared memory tiles of Y, warp-level reductions for dot products. Templated on rows_per_block, cols_per_block, nwarps.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {"X": "const T* [ne00, ...]", "Y": "const float* [ne10, ncols]", "ids": "const int* (optional)"},
            "outputs": {"dst": "float* [ne00, ncols]"},
            "dtypes": "T = f32, f16"
        },
        "reference": DEFAULT_REF,
        "variants": ["16 ncols instances: mmf-instance-ncols_{1..16}.cu"],
        "notes": "Small-ncols MM. Template instances for each ncols value 1-16."
    },
    {
        "id": "mul-mat-f-ids",
        "name": "mul_mat_f_ids",
        "source": "ggml/src/ggml-cuda/mmf.cuh:299",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Matrix multiply with expert IDs (MoE gating): multiplies selected rows of Y indexed by ids. Uses shared memory for IDs and warp reductions.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {"X": "const T*", "Y": "const float*", "ids": "const int*"},
            "outputs": {"dst": "float*"},
            "dtypes": "T = f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "MoE-gated variant of mul_mat_f."
    },
    {
        "id": "mul-mat-vec-f",
        "name": "mul_mat_vec_f",
        "source": "ggml/src/ggml-cuda/mmvf.cu:8",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Matrix-vector multiply (fp32/fp16 matrix times fp32 vector): each row's dot product computed across warp via warp_reduce_sum. Supports optional gating/fusion.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {"src0": "const T* [ne00, ne01]", "src1": "const float* [ne00] (or [ne00, ncols_dst])"},
            "outputs": {"dst": "float* [ne01] (or [ne01, ncols_dst])"},
            "dtypes": "T = f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Standard GEMV. Relatively straightforward."
    },
    {
        "id": "mul-mat-vec-q",
        "name": "mul_mat_vec_q",
        "source": "ggml/src/ggml-cuda/mmvq.cu:480",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Quantized matrix-vector multiply: dequantize packed weights on-the-fly, compute dot products per row, warp-level reduction. Supports all quant types and per-architecture parameter tables. Complex multi-warp cooperation with shared memory for intermediate results.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_sum", "dp4a"],
        "io_spec": {
            "inputs": {"src0": "const char* (quantized) [ne00, ne01]", "src1": "const float* [ne00] (or quantized q8_1)"},
            "outputs": {"dst": "float* [ne01]"},
            "dtypes": "all quant types -> f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Critical for inference when batch=1. Complex multi-warp shared memory sharing."
    },
    {
        "id": "mul-mat-vec-q-moe",
        "name": "mul_mat_vec_q_moe",
        "source": "ggml/src/ggml-cuda/mmvq.cu:707",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Quantized matrix-vector multiply for MoE (mixture of experts): routes computation through selected expert matrices indexed by expert IDs. Same structure as mul_mat_vec_q but with expert gating.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {"src0": "const char* (quantized) [ne00, ne01]", "src1": "const float*", "ids": "const int*"},
            "outputs": {"dst": "float*"},
            "dtypes": "all quant types -> f32"
        },
        "reference": DEFAULT_REF,
        "notes": "MoE variant of mmvq."
    },

    # --- Convolution kernels ---
    {
        "id": "conv2d",
        "name": "conv2d_kernel",
        "source": "ggml/src/ggml-cuda/conv2d.cu:73",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "2D convolution (im2col-based): each thread computes one output pixel by iterating over kernel window. Supports NCHW and NHWC layouts via template. Uses helper device functions for index computation.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"input": "const float* [N, C, H, W]", "kernel": "const float* [OC, IC, KH, KW]"},
            "outputs": {"output": "float* [N, OC, OH, OW]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Direct convolution (not im2col+GEMM). Straightforward nested loop."
    },
    {
        "id": "conv2d-dw",
        "name": "conv2d_dw_kernel",
        "source": "ggml/src/ggml-cuda/conv2d-dw.cu:82",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Depthwise 2D convolution: each output channel processed independently with its own kernel. Templates on data layout (whcn/cwhn). Simple nested loop per thread.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"input": "const T* [N, C, H, W]", "kernel": "const T* [C, 1, KH, KW]"},
            "outputs": {"output": "T* [N, C, OH, OW]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Depthwise convolution."
    },
    {
        "id": "conv2d-transpose",
        "name": "conv2d_transpose_kernel",
        "source": "ggml/src/ggml-cuda/conv2d-transpose.cu:5",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Transposed 2D convolution (deconvolution): scatters each input pixel across the upsampled output. Each thread handles one output position.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"input": "const float* [N, IC, H, W]", "kernel": "const float* [IC, OC, KH, KW]"},
            "outputs": {"output": "float* [N, OC, OH, OW]"},
            "dtypes": "f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Transposed convolution."
    },
    {
        "id": "conv-transpose-1d",
        "name": "conv_transpose_1d_kernel",
        "source": "ggml/src/ggml-cuda/conv-transpose-1d.cu:3",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "1D transposed convolution: scatters each input element across output positions with stride/dilation/padding.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src": "const float* [N, IC, L]", "kernel": "const float* [IC, OC, KW]"},
            "outputs": {"dst": "float* [N, OC, OL]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "1D deconvolution, simple scatter."
    },

    # --- Pooling kernels ---
    {
        "id": "pool2d-nchw",
        "name": "pool2d_nchw_kernel",
        "source": "ggml/src/ggml-cuda/pool2d.cu:4",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "2D pooling (max/avg) in NCHW layout. Each thread processes one output pixel by iterating over its pooling window and applying the specified reduction op.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src": "const float* [N, C, H, W]", "op": "max or avg"},
            "outputs": {"dst": "float* [N, C, OH, OW]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Standard pooling."
    },

    # --- Image to column kernels ---
    {
        "id": "im2col",
        "name": "im2col_kernel",
        "source": "ggml/src/ggml-cuda/im2col.cu:7",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "2D im2col: rearranges image patches into columns for efficient convolution as GEMM. Each thread processes one output column (one kernel application site).",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const float* [N, IC, IH, IW]"},
            "outputs": {"dst": "float* [N, IC*KH*KW, OH*OW]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Standard im2col."
    },
    {
        "id": "im2col-3d",
        "name": "im2col_3d_kernel",
        "source": "ggml/src/ggml-cuda/im2col.cu:120",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "3D im2col for volumetric convolution: rearranges 3D patches into columns. Each thread computes one output column.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src": "const float* [N, IC, ID, IH, IW]"},
            "outputs": {"dst": "float* [N, IC*KD*KH*KW, OD*OH*OW]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "3D variant of im2col."
    },
    {
        "id": "col2im-1d",
        "name": "col2im_1d_kernel",
        "source": "ggml/src/ggml-cuda/col2im-1d.cu:9",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "1D col2im: reverses im2col, scattering column elements back to image positions (for convolution backward).",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const float* [IC*KW, OW*N]"},
            "outputs": {"dst": "float* [N, IC, IW]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Inverse of im2col."
    },

    # --- Upscale kernels ---
    {
        "id": "upscale-f32",
        "name": "upscale_f32",
        "source": "ggml/src/ggml-cuda/upscale.cu:3",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Nearest-neighbor upscaling: replicates input pixels according to integer scale factors per dimension.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const float* [ne0, ne1, ne2, ne3]", "sf": "int[4]"},
            "outputs": {"dst": "float* [ne0*sf0, ne1*sf1, ne2*sf2, ne3*sf3]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Nearest-neighbor upscale."
    },
    {
        "id": "upscale-f32-bilinear",
        "name": "upscale_f32_bilinear",
        "source": "ggml/src/ggml-cuda/upscale.cu:25",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Bilinear interpolation upscaling: each output pixel computed as weighted average of 2 source pixels (in 1D) per dimension.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const float*", "sf": "int[4]"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Bilinear upscale."
    },
    {
        "id": "upscale-f32-bilinear-antialias",
        "name": "upscale_f32_bilinear_antialias",
        "source": "ggml/src/ggml-cuda/upscale.cu:86",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Bilinear upscale with anti-aliasing: applies low-pass filtering before downscaling to prevent aliasing. Uses weighted averaging over source pixels.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src0": "const float*"}, "outputs": {"dst": "float*"}, "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Anti-aliased bilinear."
    },
    {
        "id": "upscale-f32-bicubic",
        "name": "upscale_f32_bicubic",
        "source": "ggml/src/ggml-cuda/upscale.cu:170",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Bicubic interpolation upscaling: each output pixel uses 4 source pixels per dimension with cubic weights (alpha=-0.75, matching PyTorch). Device functions for cubic weights.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const float*", "sf": "int[4]", "pixel_offset": "float"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Bicubic upscale."
    },

    # --- Pad kernels ---
    {
        "id": "pad-f32",
        "name": "pad_f32",
        "source": "ggml/src/ggml-cuda/pad.cu:10",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Reflect/constant pad along up to 4 dimensions. Uses wrap-around index arithmetic with pad values.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src": "const float* [s00, s01, s02, s03]"},
            "outputs": {"dst": "float* [d00, d01, d02, d03]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Multi-dimensional padding."
    },
    {
        "id": "pad-reflect-1d",
        "name": "pad_reflect_1d_kernel_f32",
        "source": "ggml/src/ggml-cuda/pad_reflect_1d.cu:3",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "1D reflection padding: mirrors elements at boundaries. Each thread computes one output element with reflection logic.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src": "const float* [ne0, ne1, ne2, ne3]"},
            "outputs": {"dst": "float* [ne0+p0, ne1+p1, ...]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Reflection padding."
    },

    # --- Loss kernels ---
    {
        "id": "cross-entropy-loss-f32",
        "name": "cross_entropy_loss_f32",
        "source": "ggml/src/ggml-cuda/cross-entropy-loss.cu:9",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Cross-entropy loss: loss = -sum(log_softmax(x)[target]) / nrows. Uses warp_reduce_sum for final reduction. Supports optional logits via template.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_max", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {"src0": "const float* (logits) [nrows, ne00]", "src1": "const float* (targets) [nrows]"},
            "outputs": {"dst": "float* [1]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Classification loss with warp reductions."
    },
    {
        "id": "cross-entropy-loss-back-f32",
        "name": "cross_entropy_loss_back_f32",
        "source": "ggml/src/ggml-cuda/cross-entropy-loss.cu:53",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Cross-entropy loss backward: grad_x = softmax(x_fwd) - one_hot(target). Uses warp_reduce_sum for softmax denominator.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_max", "warp_reduce_sum"],
        "io_spec": {
            "inputs": {"grad": "const float* [nrows]", "src0f": "const float* (forward logits)", "src1f": "const float* (targets)"},
            "outputs": {"dst": "float* [nrows, ne00]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Backward of cross-entropy."
    },

    # --- Sort / top-k kernels ---
    {
        "id": "argsort-init-indices",
        "name": "init_indices",
        "source": "ggml/src/ggml-cuda/argsort.cu:12",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Initialize argsort indices: indices[col] = col for each row, used as starting permutation before sorting.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {},
            "outputs": {"indices": "int* [nrows, ncols]"},
            "dtypes": "i32"
        },
        "reference": DEFAULT_REF,
        "notes": "Simple initialization."
    },
    {
        "id": "argsort-init-offsets",
        "name": "init_offsets",
        "source": "ggml/src/ggml-cuda/argsort.cu:22",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Initialize argsort column offsets for CUB: offsets[i] = i * ncols_padded.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {},
            "outputs": {"offsets": "int* [nrows]"},
            "dtypes": "i32"
        },
        "reference": DEFAULT_REF,
        "notes": "Helper for CUB segmented argsort."
    },
    {
        "id": "argsort-f32-i32",
        "name": "k_argsort_f32_i32",
        "source": "ggml/src/ggml-cuda/argsort.cu:167",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Parallel argsort using CUB: each row sorted independently via CUB DeviceSegmentedRadixSort. Each thread block handles one row, maintaining index array in shared memory with swap-based sorting.",
        "cuda_constructs": ["__shared__", "cub"],
        "io_spec": {
            "inputs": {"x": "const float* [nrows, ncols]"},
            "outputs": {"dst": "int* [nrows, ncols] (sorted indices)"},
            "dtypes": "f32 input, i32 output"
        },
        "reference": DEFAULT_REF,
        "notes": "Depends on CUB segmented radix sort."
    },
    {
        "id": "topk-moe",
        "name": "topk_moe_cuda",
        "source": "ggml/src/ggml-cuda/topk-moe.cu:91",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Top-K selection for Mixture-of-Experts gating. Each warp processes a subset of experts, maintains K best values/indices using warp-shuffle comparisons, applies softmax/sigmoid/sqrt-softplus activation. Complex warp-level tournament search with multiple stages.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_max"],
        "io_spec": {
            "inputs": {"logits": "const float* [nrows, ncols]", "k": "int", "n_experts": "int"},
            "outputs": {"vals": "float* [nrows, k]", "indices": "int* [nrows, k]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Complex warp-level tournament sort. Hard to port due to warp-shuffle-dependent selection logic."
    },

    # --- Rope / Position encoding kernels ---
    {
        "id": "rope-norm",
        "name": "rope_norm",
        "source": "ggml/src/ggml-cuda/rope.cu:44",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Standard RoPE (Rotary Position Embedding): applies rotation to pairs of elements using sin/cos of precomputed frequencies. Supports forward/backward, multiple dtype modes, and YaRN extension for context extension.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T* [ne0, ne1, ne2, ne3]", "pos": "const float*", "freq_factors": "const float* (YaRN)", "theta_scale": "float"},
            "outputs": {"dst": "T* [same shape]"},
            "dtypes": "T = f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Standard RoPE with YaRN extension."
    },
    {
        "id": "rope-neox",
        "name": "rope_neox",
        "source": "ggml/src/ggml-cuda/rope.cu:116",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "NeoX-style RoPE: GPT-NeoX half-dimension rotation. Only the first half of dimensions get rotated (the second half is passed through unchanged).",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T*", "pos": "const float*"},
            "outputs": {"dst": "T*"},
            "dtypes": "T = f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "NeoX RoPE variant."
    },
    {
        "id": "rope-multi",
        "name": "rope_multi",
        "source": "ggml/src/ggml-cuda/rope.cu:185",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Multi-section RoPE: applies rotation with multiple sets of frequency bases (for models with different sections having different embedding sizes).",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T*", "pos": "const float*", "sections": "const int*"},
            "outputs": {"dst": "T*"},
            "dtypes": "T = f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Multi-section RoPE."
    },
    {
        "id": "rope-vision",
        "name": "rope_vision",
        "source": "ggml/src/ggml-cuda/rope.cu:271",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Vision-specific RoPE: applies position embedding to 2D visual tokens with x/y position coordinates. Each dimension pair rotated by x or y position frequency.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T*", "pos": "const float* (2D positions)"},
            "outputs": {"dst": "T*"},
            "dtypes": "T = f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "2D vision RoPE."
    },

    # --- Get rows / embedding lookup ---
    {
        "id": "get-rows",
        "name": "k_get_rows",
        "source": "ggml/src/ggml-cuda/getrows.cu:6",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Gather rows from quantized weight matrix by index: dst_row[i] = dequantize(src_row[indices[i]]). Each thread block handles one output row, dequantizing on-the-fly.",
        "cuda_constructs": ["warp-shuffle"],
        "io_spec": {
            "inputs": {"src0": "const char* (quantized) [ne00, ne01]", "src1": "const int32_t* [ne10] (indices)"},
            "outputs": {"dst": "float* [ne10, ne00]"},
            "dtypes": "quantized -> f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Embedding lookup with dequant."
    },
    {
        "id": "get-rows-kq",
        "name": "k_get_rows_kq",
        "source": "ggml/src/ggml-cuda/getrows.cu:44",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Gather rows from K-quant format (q2_K..q6_K) by index. Each thread handles a segment of the K-quant block with its complex multi-scale dequantization.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src0": "const void* (K-quant) [ne00, ne01]", "src1": "const int32_t* [ne10]"}, 
            "outputs": {"dst": "dst_t* [ne10, ne00]"},
            "dtypes": "K-quant -> f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "K-quant specific gather."
    },
    {
        "id": "get-rows-float",
        "name": "k_get_rows_float",
        "source": "ggml/src/ggml-cuda/getrows.cu:73",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Gather rows from float matrix: simple indexed copy. Each thread handles one element.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src0": "const float* [ne00, ne01]", "src1": "const int32_t* [ne10]"},
            "outputs": {"dst": "float* [ne10, ne00]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Simple float gather."
    },
    {
        "id": "get-rows-float-vec",
        "name": "k_get_rows_float_vec",
        "source": "ggml/src/ggml-cuda/getrows.cu:105",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Vectorized float row gather: uses float4 loads for coalesced memory access. Each thread handles blocks of 4 floats at a time.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src0": "const float*", "src1": "const int32_t*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32 (float4 vectorized)"
        },
        "reference": DEFAULT_REF,
        "notes": "Vectorized gather for performance."
    },
    {
        "id": "get-rows-back-float",
        "name": "k_get_rows_back_float",
        "source": "ggml/src/ggml-cuda/getrows.cu:133",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Backward of get_rows: scatter-add gradients to the source matrix at indexed positions.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src0": "const float* (grads) [ne00, ne10]", "src1": "const int32_t* [ne10]"},
            "outputs": {"dst": "float* [ne00, ne01]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Scatter-add backward."
    },

    # --- State space model kernels (MEDIUM-HIGH) ---
    {
        "id": "ssm-conv-f32",
        "name": "ssm_conv_f32",
        "source": "ggml/src/ggml-cuda/ssm-conv.cu:6",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "SSM convolution: performs 1D convolution with state-space model kernel (Mamba-style). Each block handles one batch element, using shared memory for the input tile. Supports optional SiLU activation.",
        "cuda_constructs": ["__shared__"],
        "io_spec": {
            "inputs": {"src0": "const float* (state kernel) [d_inner, d_conv]", "src1": "const float* (input) [n_tokens, n_channels, d_inner]"},
            "outputs": {"dst": "float* [n_tokens, n_channels, d_inner]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Mamba SSM convolution."
    },
    {
        "id": "ssm-conv-long-token-f32",
        "name": "ssm_conv_long_token_f32",
        "source": "ggml/src/ggml-cuda/ssm-conv.cu:61",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Long-token SSM convolution: variant optimized for long sequence processing with split n_t processing across blocks per channel. Uses external shared memory for intermediate results.",
        "cuda_constructs": ["__shared__"],
        "io_spec": {
            "inputs": {"src0": "const float*", "src1": "const float*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Long-sequence variant of SSM convolution."
    },
    {
        "id": "ssm-scan-f32",
        "name": "ssm_scan_f32",
        "source": "ggml/src/ggml-cuda/ssm-scan.cu:34",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "SSM scan (selective scan, Mamba-style): parallel prefix scan over d_state dimension using CUB block-level scan. Computes the recurrent state update: h_t = A_t * h_{t-1} + B_t * x_t, y_t = C_t * h_t. Uses complex shared memory allocation for CUB temp storage, split D and d_state dimensions.",
        "cuda_constructs": ["__shared__", "cub", "warp-shuffle"],
        "io_spec": {
            "inputs": {
                "src0-src5": "const float* (A, B, C, dt, delta_bias, D params)",
                "src6": "const int32_t* (s indices)"
            },
            "outputs": {"dst": "float* [n_tokens, n_channels, d_inner]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Most complex SSM kernel. Depends on CUB block-level scan."
    },
    {
        "id": "ssm-scan-f32-group",
        "name": "ssm_scan_f32_group",
        "source": "ggml/src/ggml-cuda/ssm-scan.cu:144",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Grouped SSM scan: processes multiple d_state groups in one block. Variant of ssm_scan_f32 for grouped state dimensions. Uses CUB block scan in shared memory.",
        "cuda_constructs": ["__shared__", "cub"],
        "io_spec": {
            "inputs": {"src0-src6": "const float* / int32_t*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Grouped variant of SSM selective scan."
    },
    {
        "id": "ssm-ssd-prepare-dt",
        "name": "ssm_ssd_prepare_dt_kernel",
        "source": "ggml/src/ggml-cuda/ssm-scan.cu:347",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "SSD SSM: prepare delta_t from A and dt parameters. Computes the discretized state transition: exp(A * dt) and B * dt, with warp-level reductions. Uses CUB block scan for prefix sum.",
        "cuda_constructs": ["__shared__", "cub", "warp-shuffle"],
        "io_spec": {
            "inputs": {"A": "const float*", "B": "const float*", "C": "const float*", "dt": "const float*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "SSD model delta preparation."
    },
    {
        "id": "ssm-ssd-pre-matmul",
        "name": "ssm_ssd_pre_matmul_kernel",
        "source": "ggml/src/ggml-cuda/ssm-scan.cu:433",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "SSD pre-matmul: computes the C * state inner product before the scan phase. Each thread block handles one chunk of the sequence, using shared memory for B and C tiles.",
        "cuda_constructs": ["__shared__"],
        "io_spec": {
            "inputs": {"B": "const float*", "C": "const float*", "state": "const float*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "SSD prep step."
    },
    {
        "id": "ssm-ssd-scale-state",
        "name": "ssm_ssd_scale_state_kernel",
        "source": "ggml/src/ggml-cuda/ssm-scan.cu:521",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "SSD scale state: element-wise scaling of the recurrent state by A-discretized factors across the sequence.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"state": "const float*", "scale": "const float*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Simple element-wise scaling."
    },
    {
        "id": "ssm-ssd-init-state",
        "name": "ssm_ssd_init_state_kernel",
        "source": "ggml/src/ggml-cuda/ssm-scan.cu:548",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "SSD init state: initialize the recurrent state from B and x (input) and dt for the first position in each sequence chunk.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"B": "const float*", "x": "const float*", "dt": "const float*"},
            "outputs": {"state": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "State initialization for SSD."
    },

    # --- Gated linear attention / Delta Net kernels ---
    {
        "id": "gated-linear-attn-f32",
        "name": "gated_linear_attn_f32",
        "source": "ggml/src/ggml-cuda/gla.cu:5",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Gated Linear Attention (GLA): recurrent linear attention with data-dependent gate. Maintains hidden state S updated as S = td * S + k * v^T, output = (S^T * q) * r. Uses shared memory for per-head K, R, TD vectors and warp reductions.",
        "cuda_constructs": ["__shared__", "warp-shuffle"],
        "io_spec": {
            "inputs": {
                "k": "const float* [B, T, C]",
                "v": "const float* [B, T, C]",
                "r": "const float* [B, T, C]",
                "td": "const float* [B, T, C]",
                "s": "const float* [B, H, C/H, C/H] (state)",
                "scale": "float"
            },
            "outputs": {"dst": "float* [B, T, C]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Recurrent linear attention. Templated on head_size (64/128)."
    },
    {
        "id": "gated-delta-net",
        "name": "gated_delta_net_cuda",
        "source": "ggml/src/ggml-cuda/gated_delta_net.cu:5",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Gated DeltaNet: recurrent delta-net update with gating. Maintains recurrent state updated via delta rule. Uses shared memory for per-head key/value vectors, launch bounds tuned to warp size.",
        "cuda_constructs": ["__shared__", "warp-shuffle"],
        "io_spec": {
            "inputs": {
                "q": "const float*", "k": "const float*", "v": "const float*",
                "g": "const float* (gate)", "beta": "const float*",
                "curr_state": "const float*"
            },
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Delta-net recurrent layer."
    },

    # --- RWKV kernels ---
    {
        "id": "rwkv-wkv-f32",
        "name": "rwkv_wkv_f32",
        "source": "ggml/src/ggml-cuda/wkv.cu:5",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "RWKV WKV attention (v5/v6): time-mixing attention with learned time-decay and token-shift. Each head computes: state *= exp(-exp(w)), state += k*v, output = (state * r). Uses shared memory for per-head K, R, TF, TD vectors.",
        "cuda_constructs": ["__shared__", "warp-shuffle"],
        "io_spec": {
            "inputs": {
                "k": "const float* [B, T, C]", "v": "const float* [B, T, C]",
                "r": "const float* [B, T, C]", "tf": "const float* [B, T, C]",
                "td": "const float* [B, T, C]", "s": "const float* [B, H, C/H, C/H] (state)"
            },
            "outputs": {"dst": "float* [B, T, C]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "RWKV time-mixing. Templated on head_size."
    },
    {
        "id": "rwkv-wkv7-f32",
        "name": "rwkv_wkv7_f32",
        "source": "ggml/src/ggml-cuda/wkv.cu:69",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "RWKV-7 attention: updated variant with separate r, w, k, v, a, b tensors. Uses learned state initialization and more complex gating. States: s = diag(w)*s + k^T * v, output = (s * r + k*a) * b.",
        "cuda_constructs": ["__shared__"],
        "io_spec": {
            "inputs": {
                "r": "const float*", "w": "const float*", "k": "const float*",
                "v": "const float*", "a": "const float*", "b": "const float*", "s": "const float*"
            },
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "RWKV-7 architecture."
    },

    # --- DSv4-HC (DeepSeek v4) kernels ---
    {
        "id": "dsv4-hc-comb-f32",
        "name": "dsv4_hc_comb_f32",
        "source": "ggml/src/ggml-cuda/dsv4-hc.cu:36",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "DeepSeek v4 head combination: fuses multiple attention head outputs with learned combination weights. Uses device helper functions for per-row/per-column normalization of combination scores.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"q": "const float*", "k": "const float*", "v": "const float*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "DSv4 multi-head combination."
    },
    {
        "id": "dsv4-hc-pre-f32",
        "name": "dsv4_hc_pre_f32",
        "source": "ggml/src/ggml-cuda/dsv4-hc.cu:103",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "DSv4 head-combination pre-processing: computes per-head QK dot products to produce combination scores before the main combination step.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"q": "const float*", "k": "const float*"},
            "outputs": {"comb": "float* (combination scores)"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Preprocessing for DSv4 attention combination."
    },
    {
        "id": "dsv4-hc-post-f32",
        "name": "dsv4_hc_post_f32",
        "source": "ggml/src/ggml-cuda/dsv4-hc.cu:140",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "DSv4 head-combination post-processing: applies combination weights to V and writes final outputs.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"comb": "const float*", "v": "const float*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Post-processing for DSv4."
    },

    # --- Quantize kernels (MEDIUM-HIGH) ---
    {
        "id": "quantize-q8-1",
        "name": "quantize_q8_1",
        "source": "ggml/src/ggml-cuda/quantize.cu:54",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Quantize float32 to q8_1 format: each block of 32 floats -> d (scale) + 32 * int8. Uses warp-shuffle for block-level amax and scale computation.",
        "cuda_constructs": ["warp-shuffle"],
        "io_spec": {
            "inputs": {"x": "const float* [ne0, ne1, ...]", "ids": "const int* (optional)"},
            "outputs": {"vy": "block_q8_1* [ne0/QK8_1, ne1, ...]"},
            "dtypes": "f32 -> q8_1"
        },
        "reference": DEFAULT_REF,
        "notes": "Block quantization with warp reductions for scale."
    },
    {
        "id": "quantize-mmq-nvfp4",
        "name": "quantize_mmq_nvfp4",
        "source": "ggml/src/ggml-cuda/quantize.cu:128",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Quantize to NVFP4 format for mmq: 4-bit floating point with block scaling and block-level fp8 (e8m0) exponent sharing. Uses warp-shuffle for amax, shared memory for warp-level scale aggregation. Complex inter-warp data exchange with __shfl_sync.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "dp4a"],
        "io_spec": {
            "inputs": {"x": "const float* [ne00, ...]", "ids": "const int* (optional)"},
            "outputs": {"vy": "block_nvfp4*"},
            "dtypes": "f32 -> nvfp4"
        },
        "reference": DEFAULT_REF,
        "notes": "Complex quantization with inter-warp shuffle for data packing. Hard to port cleanly."
    },
    {
        "id": "quantize-mmq-mxfp4",
        "name": "quantize_mmq_mxfp4",
        "source": "ggml/src/ggml-cuda/quantize.cu:338",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Quantize to MXFP4 format (microscaling fp4): 4-bit floating point with 32-element block-level e8m0 shared exponent. Uses warp-shuffle for amax reduction across warps, complex per-element rounding to fp4.",
        "cuda_constructs": ["warp-shuffle", "__shared__"],
        "io_spec": {
            "inputs": {"x": "const float*"}, "outputs": {"vy": "block_mxfp4*"},
            "dtypes": "f32 -> mxfp4"
        },
        "reference": DEFAULT_REF,
        "notes": "MXFP4 quantization with warp-level scale reduction."
    },
    {
        "id": "quantize-mmq-q8-1",
        "name": "quantize_mmq_q8_1",
        "source": "ggml/src/ggml-cuda/quantize.cu:458",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Quantize to q8_1 format for mmq use with multiple data layouts (D4, DS4, D2S6). Uses warp-shuffle for block level amax reduction.",
        "cuda_constructs": ["warp-shuffle"],
        "io_spec": {
            "inputs": {"x": "const float*"}, "outputs": {"vy": "block_q8_1_mmq*"},
            "dtypes": "f32 -> q8_1 (mmq layout)"
        },
        "reference": DEFAULT_REF,
        "variants": ["3 data layouts: D4, DS4, D2S6"],
        "notes": "mmq-specific q8_1 variant with different data layouts."
    },

    # --- Concat kernels ---
    {
        "id": "concat-cont",
        "name": "concat_cont",
        "source": "ggml/src/ggml-cuda/concat.cu:7",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Concatenate two tensors along contiguous dimension (dim 1 or 2). Copies src0 and src1 into dst with offset. Template on dimension.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T* [ne00, ne01, ne02]", "y": "const T* [ne10, ne11, ne12]"},
            "outputs": {"dst": "T* [ne0, ne1, ne2]"},
            "dtypes": "T = f32, f16, i32"
        },
        "reference": DEFAULT_REF,
        "notes": "Simple concatenation."
    },
    {
        "id": "concat-non-cont",
        "name": "concat_non_cont",
        "source": "ggml/src/ggml-cuda/concat.cu:84",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Concatenate along non-contiguous dimension (dim != 1, 2). Uses strided index arithmetic with 3D unwrapping.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src0": "const char*", "src1": "const char*"},
            "outputs": {"dst": "char*"},
            "dtypes": "generic"
        },
        "reference": DEFAULT_REF,
        "notes": "Non-contig concat path."
    },

    # --- Count equal ---
    {
        "id": "count-equal",
        "name": "count_equal",
        "source": "ggml/src/ggml-cuda/count-equal.cu:7",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Count equal elements: compares two arrays element-wise, atomically increments a counter for each match. Uses atomicAdd for cross-block accumulation.",
        "cuda_constructs": ["atomicAdd"],
        "io_spec": {
            "inputs": {"x": "const T* [dk, k]", "y": "const T* [dk, k]"},
            "outputs": {"dst": "int64_t* (count)"},
            "dtypes": "T = f32, f16, i32"
        },
        "reference": DEFAULT_REF,
        "notes": "Element-wise comparison with atomic counter."
    },

    # --- Cumsum kernels ---
    {
        "id": "cumsum-cub",
        "name": "cumsum_cub_kernel",
        "source": "ggml/src/ggml-cuda/cumsum.cu:12",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Cumulative sum using CUB block-level inclusive scan. Uses CUB BlockScan with shared memory for temp storage. Handles cross-block carry via shared memory flag.",
        "cuda_constructs": ["__shared__", "cub"],
        "io_spec": {
            "inputs": {"x": "const T* [nelements]"},
            "outputs": {"dst": "T* [nelements]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "CUB block scan. Map to sycl::joint_exclusive_scan via oneDPL."
    },
    {
        "id": "cumsum-kernel",
        "name": "cumsum_kernel",
        "source": "ggml/src/ggml-cuda/cumsum.cu:86",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Fallback cumsum: manual prefix scan using shared memory, without CUB dependency. Block-level prefix sum with cross-block carry propagation via external shared memory.",
        "cuda_constructs": ["__shared__"],
        "io_spec": {
            "inputs": {"x": "const float* [nelements]"},
            "outputs": {"dst": "float* [nelements]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Manual prefix scan, no CUB dependency."
    },

    # --- Lightning Indexer kernels ---
    {
        "id": "lightning-indexer-wmma",
        "name": "lightning_indexer_kernel_wmma",
        "source": "ggml/src/ggml-cuda/lightning-indexer.cu:19",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Lightning attention indexer using WMMA (Tensor Core). Computes Q*K^T per batch/stream with WMMA matrix multiply for each head. Complex shared memory layout with separate tiles for Q, K (quantized or fp16), and QK results. Multi-warp cooperation for inner loops over WMMA tiles.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "wmma"],
        "io_spec": {
            "inputs": {
                "Q": "const float* [n_batch, n_stream, n_head, N_EMBD]",
                "K": "const char* (quantized/fp16) [n_batch, n_kv, n_head, N_EMBD]",
                "W": "const float* [n_head, n_head]",
                "M": "const half* (mask)"
            },
            "outputs": {"dst": "float* [n_batch, n_stream, n_kv, n_head]"},
            "dtypes": "f32, quantized K, f16 mask"
        },
        "reference": DEFAULT_REF,
        "notes": "WMMA-based attention indexer. Requires WMMA -> SYCL joint_matrix translation. Available only on Turing+ (TURING_MMA_AVAILABLE)."
    },
    {
        "id": "lightning-indexer-vec",
        "name": "lightning_indexer_kernel_vec",
        "source": "ggml/src/ggml-cuda/lightning-indexer.cu:244",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Lightning attention indexer (vector path, no WMMA): each warp processes one K vector, computing Q*K dot products using float4 loads and shared memory for Q tiles. Falls back to element-wise dot product without Tensor Core acceleration.",
        "cuda_constructs": ["__shared__", "warp-shuffle"],
        "io_spec": {
            "inputs": {"Q": "const float*", "K": "const char* (quantized/fp16)", "W": "const float*", "M": "const half*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Non-WMMA fallback for lightning attention. Simpler to port."
    },

    # --- Argmax kernel ---
    {
        "id": "argmax-f32",
        "name": "argmax_f32",
        "source": "ggml/src/ggml-cuda/argmax.cu:8",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Argmax per column (row-wise argmax): each warp handles one column, using warp-shuffle for maximum tracking and index propagation. Multi-warp case uses shared memory for inter-warp reduction.",
        "cuda_constructs": ["warp-shuffle", "__shared__", "warp_reduce_max"],
        "io_spec": {
            "inputs": {"x": "const float* [ncols, nrows]"},
            "outputs": {"dst": "int32_t* [ncols]"},
            "dtypes": "f32 input, i32 output"
        },
        "reference": DEFAULT_REF,
        "notes": "Warp-level argmax reduction."
    },

    # --- All-reduce kernels (multi-GPU) ---
    {
        "id": "allreduce-ar-kernel",
        "name": "ggml_cuda_ar_kernel",
        "source": "ggml/src/ggml-cuda/allreduce.cu:109",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "GPU-to-GPU all-reduce via NVLink/shared memory signal-based protocol. Coordinates multiple GPUs reading from and writing to peer GPU memory using device-level signaling with ld.global.acquire/st.global.release semantics. Requires multi-GPU setup.",
        "cuda_constructs": ["__shared__", "atomicAdd", "cooperative-groups (grid sync)", "peer memory access"],
        "io_spec": {
            "inputs": {"data": "T_dst* [local]"},
            "outputs": {"data": "T_dst* (reduced across GPUs)"},
            "dtypes": "f32, f16, bf16"
        },
        "reference": DEFAULT_REF,
        "notes": "Multi-GPU all-reduce. Uses ld.global.acquire/st.global.release which are CUDA-specific. Hardest to port - skip for single-GPU SYCL target."
    },
    {
        "id": "allreduce-ar-add-kernel",
        "name": "ggml_cuda_ar_add_kernel",
        "source": "ggml/src/ggml-cuda/allreduce.cu:207",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Post-allreduce addition: adds reduced data back to output buffer after the all-reduce completes. Simple element-wise addition.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"data": "const T_src* [ne] (reduced)"},
            "outputs": {"data": "T_dst* [ne] (in-place add)"},
            "dtypes": "f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Post-reduce add. Simple but only useful with all-reduce."
    },

    # --- Infrastructure / meta kernels ---
    {
        "id": "compute-batched-ptrs",
        "name": "k_compute_batched_ptrs",
        "source": "ggml/src/ggml-cuda/ggml-cuda.cu:1335",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Compute batched pointer arrays: maps linear indices to device pointers for batched kernel launches. Each thread computes one pointer using stride arithmetic.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"base_ptrs": "metadata"},
            "outputs": {"batched_ptrs": "void** [n_batches]"},
            "dtypes": "ptr arithmetic"
        },
        "reference": DEFAULT_REF,
        "notes": "Infrastructure kernel for batched ops."
    },
    {
        "id": "compute-out-prod-ptrs",
        "name": "k_compute_out_prod_ptrs",
        "source": "ggml/src/ggml-cuda/out-prod.cu:5",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Compute pointers for batched outer product: sets up device pointer arrays for cuBLAS batched gemm calls.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"src0": "const float*", "src1": "const float*"},
            "outputs": {"ptr_arrays": "float** [batched]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Precompute pointer arrays for cuBLAS batched gemm."
    },
    {
        "id": "mm-ids-helper",
        "name": "mm_ids_helper",
        "source": "ggml/src/ggml-cuda/mmid.cu:28",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Helper kernel for MoE matrix-multiply: computes expert ID compaction using warp-shuffle and shared memory. Each warp processes a subset of expert IDs, compacting them with prefix-sum style accumulation.",
        "cuda_constructs": ["warp-shuffle", "__shared__"],
        "io_spec": {
            "inputs": {"ids": "const int*"},
            "outputs": {"compacted_ids": "int*"},
            "dtypes": "i32"
        },
        "reference": DEFAULT_REF,
        "notes": "MoE ID compaction helper. Uses warp-level prefix sum."
    },
    {
        "id": "opt-step-sgd-f32",
        "name": "opt_step_sgd_f32",
        "source": "ggml/src/ggml-cuda/opt-step-sgd.cu:6",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "SGD optimizer step: x -= lr * g (with optional weight decay). Element-wise parameter update.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "float* [k]", "g": "const float* [k] (gradient)", "pars": "struct (lr, wd)"},
            "outputs": {"x": "float* (updated in-place)"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Simple element-wise SGD update."
    },
    {
        "id": "opt-step-adamw-f32",
        "name": "opt_step_adamw_f32",
        "source": "ggml/src/ggml-cuda/opt-step-adamw.cu:6",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "AdamW optimizer step: maintains m (first moment) and v (second moment) buffers, applies bias correction, weight decay, and Adam update. Element-wise with per-element sqrt division.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {
                "x": "float* [k]", "g": "const float* [k]",
                "g_m": "float* [k] (momentum buffer)", "g_v": "float* [k] (velocity buffer)",
                "pars": "struct (lr, beta1, beta2, eps, wd, n_iter)"
            },
            "outputs": {"x, g_m, g_v": "float* (updated in-place)"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Standard AdamW optimizer step."
    },
    {
        "id": "get-batch-pointers",
        "name": "get_batch_pointers",
        "source": "ggml/src/ggml-cuda/solve_tri.cu:8",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Precompute batch pointers for triangular solve: maps linear batch indices to device pointers for batched cuBLAS trsm calls.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"A": "const float*", "X": "const float*"},
            "outputs": {"A_ptrs_dev": "float**", "X_ptrs_dev": "float**"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Pointer setup for batched cuBLAS."
    },
    {
        "id": "solve-tri-f32-fast",
        "name": "solve_tri_f32_fast",
        "source": "ggml/src/ggml-cuda/solve_tri.cu:92",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Triangular solve (small matrix): in-place Gaussian elimination for small triangular systems (n <= 8). Uses shared memory for the A matrix, cooperative warp-level solve with __shfl_sync for pivot exchange. Falls back to cuBLAS for larger systems.",
        "cuda_constructs": ["__shared__", "warp-shuffle"],
        "io_spec": {
            "inputs": {"A": "const float* [n, n] (triangular)", "B": "const float* [n, nrhs]", "n": "int (<= 8)"},
            "outputs": {"X": "float* [n, nrhs]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Small-n triangular solve with warp cooperativeness. For larger n, falls back to cuBLAS."
    },

    # --- Kernel launches in ggml-cuda.cu ---
    {
        "id": "top-k",
        "name": "top_k (via CUB)",
        "source": "ggml/src/ggml-cuda/top-k.cu",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Top-K selection using CUB device-level radix sort. Sorts by value, selects top K, extracts values and indices. Used for standard (non-MoE) top-k logit selection.",
        "cuda_constructs": ["cub"],
        "io_spec": {
            "inputs": {"x": "const float* [ncols]"},
            "outputs": {"vals": "float* [k]", "indices": "int* [k]"},
            "dtypes": "f32, i32"
        },
        "reference": DEFAULT_REF,
        "notes": "CUB-based top-k. SYCL has oneDPL sort equivalent."
    },
]

def main():
    index_kernels = []
    counts = {"low": 0, "medium": 0, "high": 0}

    for kdef in KERNELS:
        kid = kdef["id"]
        risk = kdef["risk"]
        counts[risk] += 1

        # Write detail file
        detail = dict(kdef)
        detail["updated_at"] = now_ts
        detail_path = os.path.join(KERNELS_DIR, f"{kid}.json")
        with open(detail_path, "w") as f:
            json.dump(detail, f, indent=2)
            f.write("\n")

        # Add to index
        index_kernels.append({
            "id": kid,
            "source": kdef["source"],
            "source_lang": kdef.get("source_lang", "cuda"),
            "target_style": kdef.get("target_style", "plain-sycl"),
            "status": kdef["status"],
            "risk": risk,
            "detail": f"kernels/{kid}.json",
            "notes": kdef.get("notes", "")
        })

    # Write index.json
    index = {
        "schema": "kernel-index",
        "kernels": index_kernels,
        "updated_at": now_ts
    }
    index_path = os.path.join(KERNELS_DIR, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    print(f"Written {len(index_kernels)} kernel entries to {KERNELS_DIR}")
    print(f"Risk distribution: low={counts['low']}, medium={counts['medium']}, high={counts['high']}")
    return counts

if __name__ == "__main__":
    main()