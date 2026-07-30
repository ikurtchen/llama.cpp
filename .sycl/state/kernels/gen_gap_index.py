#!/usr/bin/env python3
"""Generate gap-focused kernel index and detail files for the truly missing CUDA kernels."""

import json, os
from datetime import datetime, timezone

ROOT = "/localdisk/kurt/workspace/code/ai_coding/copilot/llama.cpp"
KERNELS_DIR = os.path.join(ROOT, ".sycl", "state", "kernels")
os.makedirs(KERNELS_DIR, exist_ok=True)
now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

DEFAULT_REF = {"kind": "cpu", "notes": "llama.cpp CPU (ggml) backend produces the reference output"}

GAP_KERNELS = [
    {
        "id": "allreduce-ar-kernel",
        "name": "ggml_cuda_ar_kernel",
        "source": "ggml/src/ggml-cuda/allreduce.cu:109",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "GPU-to-GPU all-reduce via NVLink peer-memory signal protocol. Coordinates multiple GPUs using device-level semaphores with ld.global.acquire/st.global.release semantics.",
        "cuda_constructs": ["cooperative-groups", "__shared__", "atomicAdd", "peer memory access"],
        "io_spec": {
            "inputs": {"data": "T_dst* (local GPU buffer)"},
            "outputs": {"data": "T_dst* (reduced across all GPUs)"},
            "dtypes": "f32, f16, bf16"
        },
        "reference": DEFAULT_REF,
        "notes": "Multi-GPU only. Uses ld.global.acquire/st.global.release which are CUDA-specific. Skip for single-GPU SYCL builds; needs SYCL multi-device approach if multi-GPU target required."
    },
    {
        "id": "allreduce-ar-add-kernel",
        "name": "ggml_cuda_ar_add_kernel",
        "source": "ggml/src/ggml-cuda/allreduce.cu:207",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Post-allreduce addition: adds reduced data back to output buffer. Simple element-wise vector add.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"data": "const T_src* (reduced peer data)"},
            "outputs": {"data": "T_dst* (in-place accumulate)"},
            "dtypes": "f32, f16"
        },
        "reference": DEFAULT_REF,
        "notes": "Only useful together with allreduce-ar-kernel. Can be trivially implemented if allreduce is needed."
    },
    {
        "id": "dsv4-hc-comb-f32",
        "name": "dsv4_hc_comb_f32",
        "source": "ggml/src/ggml-cuda/dsv4-hc.cu:36",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "DeepSeek v4 head combination: fuses multiple attention head outputs using learned combination weights with per-row/per-column normalization.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"q": "const float*", "k": "const float*", "v": "const float*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "DSv4 model-specific. Used by DSv4-family models."
    },
    {
        "id": "dsv4-hc-pre-f32",
        "name": "dsv4_hc_pre_f32",
        "source": "ggml/src/ggml-cuda/dsv4-hc.cu:103",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "DSv4 pre-processing: computes per-head QK dot products for combination scores.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"q": "const float*", "k": "const float*"},
            "outputs": {"comb": "float* (combination scores)"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Preprocessing step for DSv4 attention combination."
    },
    {
        "id": "dsv4-hc-post-f32",
        "name": "dsv4_hc_post_f32",
        "source": "ggml/src/ggml-cuda/dsv4-hc.cu:140",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "DSv4 post-processing: applies combination weights to V and writes final output.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"comb": "const float*", "v": "const float*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Post-processing for DSv4 head combination."
    },
    {
        "id": "fwht",
        "name": "fwht_cuda",
        "source": "ggml/src/ggml-cuda/fwht.cu:6",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Fast Walsh-Hadamard Transform: recursive butterfly network using warp shuffle for data exchange. Used as a pre-processing hint (GGML_HINT_SRC0_IS_HADAMARD).",
        "cuda_constructs": ["warp-shuffle"],
        "io_spec": {
            "inputs": {"src": "const float* [n_rows, ne]", "scale": "float"},
            "outputs": {"dst": "float* [same shape]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Uses warp shuffle for butterfly exchange. Map to sycl::group_broadcast/shuffle. Used by some quantization schemes."
    },
    {
        "id": "compute-batched-ptrs",
        "name": "k_compute_batched_ptrs",
        "source": "ggml/src/ggml-cuda/ggml-cuda.cu:1335",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Compute batched pointer arrays: maps linear indices to device pointers for batched kernel launches. Infrastructure kernel.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"base_ptrs": "metadata struct"},
            "outputs": {"batched_ptrs": "void** [n_batches]"},
            "dtypes": "pointer arithmetic"
        },
        "reference": DEFAULT_REF,
        "notes": "Infrastructure kernel for batched operations. May be handled differently in SYCL via explicit SYCL kernel batch semantics."
    },
    {
        "id": "lightning-indexer-wmma",
        "name": "lightning_indexer_kernel_wmma",
        "source": "ggml/src/ggml-cuda/lightning-indexer.cu:19",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "high",
        "algorithm": "Lightning attention Q*K^T indexer using WMMA/Tensor Core. Multi-warp cooperation with shared memory tiles for Q, K (quantized/fp16), and QK results. Complex inner loops over WMMA tiles.",
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
        "notes": "Hardest gap kernel. WMMA -> SYCL joint_matrix translation needed. Available only on Turing+."
    },
    {
        "id": "lightning-indexer-vec",
        "name": "lightning_indexer_kernel_vec",
        "source": "ggml/src/ggml-cuda/lightning-indexer.cu:244",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "medium",
        "algorithm": "Lightning attention indexer vector path (non-WMMA fallback). Each warp processes one K vector with float4 loads and shared memory Q tiles. Element-wise dot product.",
        "cuda_constructs": ["__shared__", "warp-shuffle"],
        "io_spec": {
            "inputs": {"Q": "const float*", "K": "const char* (quantized/fp16)", "W": "const float*", "M": "const half*"},
            "outputs": {"dst": "float*"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Non-WMMA fallback for lightning attention. Simpler to port than the WMMA version."
    },
    {
        "id": "opt-step-sgd-f32",
        "name": "opt_step_sgd_f32",
        "source": "ggml/src/ggml-cuda/opt-step-sgd.cu:6",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "SGD optimizer step: element-wise x -= lr * g (with optional weight decay). Training-only kernel.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "float* [k]", "g": "const float* [k] (gradient)"},
            "outputs": {"x": "float* (updated in-place)"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Training-only kernel. Trivial element-wise update. Not needed for inference."
    },
    {
        "id": "opt-step-adamw-f32",
        "name": "opt_step_adamw_f32",
        "source": "ggml/src/ggml-cuda/opt-step-adamw.cu:6",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "AdamW optimizer step: maintains m/v buffers, bias correction, weight decay. Element-wise with per-element sqrt division. Training-only kernel.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {
                "x": "float* [k]", "g": "const float* [k]",
                "g_m": "float* [k]", "g_v": "float* [k]"
            },
            "outputs": {"x, g_m, g_v": "float* (updated in-place)"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Training-only kernel. Standard AdamW element-wise formula. Not needed for inference-only deployment."
    },
    {
        "id": "snake-fused",
        "name": "snake_kernel (fused)",
        "source": "ggml/src/ggml-cuda/snake.cu:9",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Snake activation (fused): x + (1/theta) * sin^2(theta * x). CUDA implements this as a fused version of MUL->SIN->SQR->MUL->ADD chain. Also has backward pass for training.",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const T* [k]", "a": "const T*", "inv_b": "const T*"},
            "outputs": {"dst": "T* [k]"},
            "dtypes": "T = f32, f16, bf16"
        },
        "reference": DEFAULT_REF,
        "notes": "Fused kernel replacing 5-element chain. Equivalent to standard element-wise ops but fused for performance. Can be emulated via 5 separate element-wise ops (already in SYCL)."
    },
    {
        "id": "softcap-f32",
        "name": "softcap_f32",
        "source": "ggml/src/ggml-cuda/softcap.cu:3",
        "source_lang": "cuda",
        "target_style": "plain-sycl",
        "status": "pending",
        "risk": "low",
        "algorithm": "Logit softcapping: dst[i] = softcap * tanh(x[i] / softcap) * scale. Prevents logits from growing unboundedly. Also used inside flash attention (which is already covered by SYCL).",
        "cuda_constructs": [],
        "io_spec": {
            "inputs": {"x": "const float* [k]", "softcap": "float", "scale": "float"},
            "outputs": {"dst": "float* [k]"},
            "dtypes": "f32"
        },
        "reference": DEFAULT_REF,
        "notes": "Simple element-wise transcendentals. Also already handled inside flash-attention path in SYCL. Standalone kernel needed only if softcap is used outside flash attention."
    },
]

def main():
    # Remove old detail files
    import glob
    for f in glob.glob(os.path.join(KERNELS_DIR, "*.json")):
        os.remove(f)
    
    index_kernels = []
    counts = {"low": 0, "medium": 0, "high": 0}
    
    for kdef in GAP_KERNELS:
        kid = kdef["id"]
        risk = kdef["risk"]
        counts[risk] += 1
        
        detail = dict(kdef)
        detail["updated_at"] = now_ts
        detail_path = os.path.join(KERNELS_DIR, f"{kid}.json")
        with open(detail_path, "w") as f:
            json.dump(detail, f, indent=2)
            f.write("\n")
        
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
    
    index = {
        "schema": "kernel-index",
        "kernels": index_kernels,
        "updated_at": now_ts
    }
    index_path = os.path.join(KERNELS_DIR, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
    
    print(f"Written {len(index_kernels)} gap kernel entries (9 logical ops)")
    print(f"Risk: low={counts['low']}, medium={counts['medium']}, high={counts['high']}")
    return counts

if __name__ == "__main__":
    main()
