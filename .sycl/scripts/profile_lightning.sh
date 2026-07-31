#!/bin/bash
set -e

UNITRACE=/workspace/intel_gpu_tools/applications.analyzers.profilingtoolsinterfaces.sdk/tools/unitrace/build/unitrace
BUILD_DIR=/workspace/llama.cpp/build
OUTDIR=/workspace/llama.cpp/.sycl/reports/profile_lightning
rm -rf $OUTDIR && mkdir -p $OUTDIR

# Profile VectorEngineStalls first
echo "=== Collecting VectorEngineStalls ==="
$UNITRACE \
  --call-logging \
  --metric-query \
  -g VectorEngineStalls \
  --output-dir-path $OUTDIR/vec_stalls \
  -- \
  $BUILD_DIR/bin/test-backend-ops -b SYCL0 -o LIGHTNING_INDEXER -p 'nb=2048.*type_K=f16' perf 2>&1 | tail -5
echo "VectorEngineStalls done"

# Profile ComputeBasic
echo "=== Collecting ComputeBasic ==="
$UNITRACE \
  --call-logging \
  --metric-query \
  -g ComputeBasic \
  --output-dir-path $OUTDIR/comp_basic \
  -- \
  $BUILD_DIR/bin/test-backend-ops -b SYCL0 -o LIGHTNING_INDEXER -p 'nb=2048.*type_K=f16' perf 2>&1 | tail -5
echo "ComputeBasic done"

ls -la $OUTDIR/vec_stalls/ $OUTDIR/comp_basic/
