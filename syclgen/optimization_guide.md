# SYCL Kernel Optimization Guide for Intel GPUs

Reference: Intel oneAPI GPU Optimization Guide (2025.0)
https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/

This document summarizes key optimization strategies for SYCL kernels targeting
Intel Xe/Xe2 GPUs. It is used as context by the optimization phase of SyclGen.

---

## 1. Kernel Launch Configuration

### 1.1 Work-Group Size
- **Always use `nd_range`** instead of `range` for kernels that need tuning.
  `range` lets the runtime choose work-group size, which may be suboptimal.
- **Preferred sizes**: 128, 256, or 512 for Intel GPUs (multiples of sub-group size).
- **Avoid** work-group sizes < 64 (under-utilizes hardware threads) or > 1024 (exceeds limit).
- Work-group size must be a multiple of the sub-group size (8, 16, or 32).

```cpp
// BAD: runtime-chosen work-group size
q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { ... });

// GOOD: explicit nd_range with tuned work-group size
q.parallel_for(sycl::nd_range<1>({N_rounded}, {256}), [=](sycl::nd_item<1> item) { ... });
```

### 1.2 Sub-Group Size
- Request specific sub-group sizes using `[[intel::reqd_sub_group_size(16)]]`.
- SIMD16 is the native width for Xe2.
- Sub-group operations (shuffles, broadcasts, reductions) are single-cycle on Intel GPUs.

```cpp
q.parallel_for(nd_range<1>({N}, {256}),
  [=](nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
    auto sg = item.get_sub_group();
    // sub-group shuffle is very fast
    float val = sycl::group_broadcast(sg, x, 0);
  });
```

---

## 2. Memory Access Optimization

### 2.1 Global Memory Coalescing
- Adjacent work-items (within a sub-group) should access adjacent memory addresses.
- **Coalesced**: work-item `i` accesses `data[i]` → single memory transaction.
- **Strided**: work-item `i` accesses `data[i * stride]` → multiple transactions, wastes bandwidth.

```cpp
// BAD: strided access (column-major on row-major data)
float val = matrix[item.get_local_id(0) * cols + col];

// GOOD: coalesced access (row-major on row-major data)
float val = matrix[row * cols + item.get_local_id(0)];
```

### 2.2 Vectorized Memory Access
- Use `sycl::vec<T, N>` for vector loads/stores (N = 2, 4, 8, 16).
- Alternatively, use `sycl::multi_ptr` with `__attribute__((opencl_global))` for optimized loads.
- Vector loads reduce the number of memory transactions.

```cpp
// Scalar load: 4 separate transactions
float a = data[i*4], b = data[i*4+1], c = data[i*4+2], d = data[i*4+3];

// Vector load: 1 transaction
sycl::vec<float, 4> v = *reinterpret_cast<const sycl::vec<float, 4>*>(&data[i*4]);
```

### 2.3 Shared Local Memory (SLM)
- Use `sycl::local_accessor<T>` for data reuse within a work-group.
- **Keep SLM usage ≤ 64 KB** per work-group to avoid spilling.
- Use barriers (`sycl::group_barrier`) after writing to SLM before reading.
- Avoid SLM bank conflicts: 16 banks, 4-byte granularity.

```cpp
// Tiled matrix multiply using SLM
sycl::local_accessor<float, 2> tileA({TILE, TILE}, h);
sycl::local_accessor<float, 2> tileB({TILE, TILE}, h);
```

### 2.4 Prefetching
- Use `sycl::joint_group_prefetch` or explicit prefetch built-ins to hide memory latency.

```cpp
// Prefetch next iteration's data
sycl::joint_group_prefetch(item.get_group(), &data[next_offset], sycl::memory_order::relaxed);
```

---

## 3. Compute Optimization

### 3.1 Reduce Register Pressure
- Avoid large local arrays — they consume registers.
- Use smaller data types when possible (half, int16 instead of float, int32).
- Compiler may spill registers to SLM or global memory if pressure is too high.

### 3.2 Use Fast Math
- Use `-ffast-math` or `sycl::native` math functions for approximate but faster results.

```cpp
float result = sycl::native::exp(x);    // faster than sycl::exp(x)
float result = sycl::native::rsqrt(x);  // faster than 1.0f / sycl::sqrt(x)
```

### 3.3 Loop Unrolling
- Use `#pragma unroll` or `[[intel::loop_count(N)]]` for critical inner loops.

```cpp
#pragma unroll 4
for (int i = 0; i < K; i += 4) {
    acc += a[i] * b[i];
    acc += a[i+1] * b[i+1];
    acc += a[i+2] * b[i+2];
    acc += a[i+3] * b[i+3];
}
```

### 3.4 Sub-Group Operations for Reductions
- Use sub-group collective operations instead of SLM-based reductions.
- `sycl::reduce_over_group(sg, val, sycl::plus<>())` is highly efficient.

```cpp
// BAD: SLM-based reduction
local_data[lid] = val;
barrier(item.get_group());
if (lid == 0) for (int i = 0; i < WG; i++) sum += local_data[i];

// GOOD: sub-group + work-group reduction
float sg_sum = sycl::reduce_over_group(item.get_sub_group(), val, sycl::plus<float>());
float wg_sum = sycl::reduce_over_group(item.get_group(), sg_sum, sycl::plus<float>());
```

### 3.5 Joint Matrix (XMX) for Matrix Operations
- Use `sycl::ext::oneapi::experimental::matrix::joint_matrix` for matrix multiply.
- Maps directly to Intel XMX (matrix extension) hardware.
- Supported types: fp16→fp32, bf16→fp32, int8→int32, tf32→fp32.

```cpp
using namespace sycl::ext::oneapi::experimental::matrix;
joint_matrix<sub_group, float, use::accumulator, M, N> acc;
joint_matrix<sub_group, half, use::a, M, K, layout::row_major> ma;
joint_matrix<sub_group, half, use::b, K, N, layout::col_major> mb;
joint_matrix_load(sg, ma, pA, K);
joint_matrix_load(sg, mb, pB, N);
joint_matrix_mad(sg, acc, ma, mb, acc);
```

---

## 4. Occupancy Optimization

### 4.1 Thread Occupancy
- More concurrent threads = better latency hiding.
- Factors that reduce occupancy:
  - High register usage (fewer threads fit per XVE)
  - Large SLM allocation (fewer work-groups fit per Xe-core)
  - Large work-group size (fewer work-groups can run concurrently)

### 4.2 Balancing Occupancy vs. Resources
- If kernel is **compute-bound**: modest register usage, prefer higher occupancy.
- If kernel is **memory-bound**: maximize memory-level parallelism via higher occupancy.
- If kernel uses **SLM heavily**: may need to accept lower occupancy.

### 4.3 Compiler Flags
- `-ftarget-register-alloc-mode=auto` — let compiler balance registers/occupancy.
- `-fno-sycl-dead-args-optimization` — disable if it causes issues.
- Use `IGC_EnableOCLSIMD32=1` to hint for SIMD32 if beneficial.

---

## 5. Data Type Optimization

### 5.1 Use Half Precision (FP16)
- FP16 compute is 2x faster than FP32 on Xe2.
- Use `sycl::half` for intermediate computations where precision is acceptable.
- Quantized models (Q4, Q8) benefit from half-precision accumulation.

### 5.2 Use BFloat16
- Supported natively on Xe2 XMX engines.
- Good for ML inference workloads — wider dynamic range than FP16.

---

## 6. Kernel Fusion and Launch Overhead

### 6.1 Kernel Fusion
- Merge multiple small kernels into one to reduce launch overhead.
- Especially beneficial when kernels share the same data (reduces memory traffic).

### 6.2 Avoid Unnecessary Synchronization
- Minimize `queue.wait()` calls between dependent kernels.
- Use SYCL events and dependencies for asynchronous execution.
- Use in-order queues when operations are naturally sequential.

---

## 7. Common CUDA-to-SYCL Anti-Patterns to Fix

### 7.1 Missing nd_range
CUDA always specifies grid/block. A 1:1 migration often uses `sycl::range` (basic parallel_for)
which prevents work-group size tuning. Always convert to `nd_range`.

### 7.2 Over-use of Atomics
CUDA `atomicAdd` translated to `sycl::atomic_ref` is correct but slow.
Prefer sub-group/work-group reductions before atomics.

### 7.3 Literal Translation of Thread Indexing
`blockIdx.x * blockDim.x + threadIdx.x` maps to `item.get_global_id(0)` directly.
Don't compute it manually from `get_group()` and `get_local_id()` unless needed.

### 7.4 Missing Vectorization Opportunities
CUDA code often processes one element per thread. On Intel GPUs, processing
2-4 elements per thread with vector types yields better bandwidth utilization.

### 7.5 Wrong Work-Group Size for Intel GPUs
CUDA often uses 128 or 256 threads per block optimized for NVIDIA warp size (32).
Intel GPUs may benefit from different sizes (256, 512) based on SIMD16.

### 7.6 Missing Sub-Group Utilization
CUDA warp-level primitives (`__shfl_*`) translated to SLM operations instead of
`sycl::sub_group` operations. Sub-group ops are much faster.

---

## 8. Profiling Markers (Future Use)

When profiler-guided optimization is added later, use these APIs:
- **Intel VTune**: ITT API markers (`__itt_task_begin`, `__itt_task_end`)
- **Level Zero Metrics**: Hardware performance counters
- **SYCL event profiling**: `event.get_profiling_info<info::event_profiling::command_start>()`

---

## 9. Optimization Checklist

For each kernel being optimized, verify:

- [ ] Uses `nd_range` with explicit work-group size
- [ ] Work-group size is a multiple of sub-group size (16 for Xe2)
- [ ] Global memory accesses are coalesced
- [ ] Uses vector loads where applicable
- [ ] SLM usage fits within 64 KB per Xe-core
- [ ] Uses sub-group operations instead of SLM reductions where possible
- [ ] Reductions use `reduce_over_group` instead of manual loops
- [ ] No unnecessary global barriers or synchronization
- [ ] Loop unrolling applied to hot inner loops
- [ ] Register pressure is reasonable (not spilling)
- [ ] Uses appropriate data types (half where possible)
- [ ] No CUDA-to-SYCL anti-patterns present
