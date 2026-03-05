# Intel Arc Pro B60 — Target Hardware Specification

Reference: https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html

## Architecture Overview

| Property | Value |
|----------|-------|
| **Code Name** | Battlemage |
| **Microarchitecture** | Xe2 |
| **Lithography** | TSMC N5 |

## GPU Compute Specifications

| Property | Value |
|----------|-------|
| **Xe-cores** | 20 |
| **Render Slices** | 5 |
| **Xe Vector Engines (XVE)** | 160 |
| **Intel XMX Engines** | 160 |
| **Ray Tracing Units** | 20 |
| **Graphics Clock** | 2400 MHz |
| **Graphics Clock (LP)** | 2000 MHz |
| **FP32 TFLOPS** | 12.28 |
| **Int8 TOPS** | 197 |
| **TBP** | 200 W |

## Memory Specifications

| Property | Value |
|----------|-------|
| **Memory Size** | 24 GB GDDR6 |
| **Memory Interface** | 192-bit |
| **Memory Bandwidth** | 456 GB/s |
| **Memory Speed** | 19 Gbps |

## Xe2 Architecture Details

### Execution Unit Hierarchy
- **1 Xe-core** = 8 XVE (Xe Vector Engines) + 8 XMX (matrix engines)
- **1 Render Slice** = 4 Xe-cores
- **B60 Total** = 5 Render Slices × 4 Xe-cores = 20 Xe-cores = 160 XVEs

### Sub-group (SIMD) Sizes
- Supported sub-group sizes: **8, 16, 32**
- Preferred sub-group size: **16** (SIMD16 is the native width for Xe2)
- XMX operations use SIMD16 natively

### Local Memory (SLM — Shared Local Memory)
- **64 KB SLM per Xe-core** (shared across all XVEs in the Xe-core)
- SLM is banked: 16 banks × 4 bytes wide
- Avoid bank conflicts by using stride patterns that hit different banks

### Register File
- **512 bytes of GRF (General Register File) per thread** in SIMD16
- Large register mode: can use up to 256 registers (reduces occupancy)
- Small register mode: 128 registers (higher occupancy)

### Caches
- **L1 cache**: 256 KB per Xe-core (shared instruction + data)
- **L2 cache**: Shared across render slice
- **L3 cache**: Shared across GPU

### Hardware Thread Occupancy
- Each XVE can run up to **8 hardware threads** simultaneously
- Total threads per Xe-core: 8 XVEs × 8 threads = 64 threads
- Total threads on B60: 20 Xe-cores × 64 = **1280 hardware threads**
- Higher occupancy hides memory latency better
- Register pressure reduces occupancy

## Key Performance Numbers for Optimization

| Metric | Value | Notes |
|--------|-------|-------|
| Peak FP32 compute | 12.28 TFLOPS | Theoretical peak |
| Peak memory BW | 456 GB/s | GDDR6 theoretical |
| Compute intensity threshold | ~27 FP32 ops/byte | Ops below this are memory-bound |
| Max work-group size | 1024 | Hardware limit |
| Preferred work-group size | 256 or 512 | Balance occupancy vs register usage |
| Max sub-groups per work-group | 64 | 1024 / 16 (SIMD16) |
| SLM per Xe-core | 64 KB | Shared across all work-groups on that Xe-core |

## Supported Technologies

- oneAPI / SYCL (DPC++)
- OpenCL 3.0
- Level Zero
- Vulkan 1.3
- DirectX 12 Ultimate
- Intel Extension for PyTorch (IPEX)
- OpenVINO

## Optimization Implications

1. **Work-group sizing**: Use multiples of 16 (SIMD16). 256 is a good default.
2. **Sub-group operations**: Use `sycl::sub_group` for warp-like shuffles — native SIMD16 is fast.
3. **Memory coalescing**: Adjacent work-items should access adjacent memory addresses.
4. **SLM usage**: 64 KB per Xe-core — keep SLM allocation under this to avoid spilling.
5. **Vectorized loads**: Use `sycl::vec<T,N>` for 2/4/8-wide loads to maximize bandwidth.
6. **XMX for matmul**: Use `joint_matrix` SYCL extension for matrix operations on XMX engines.
7. **Occupancy**: Keep register usage low to allow more threads per XVE.
8. **Avoid divergence**: SIMD16 means 16 work-items execute in lockstep — divergent branches waste lanes.
