#!/usr/bin/env bash
# Discover Triton kernels and Triton-language usage in a project. Read-only; prints a summary.
# Usage: scan_triton.sh [project-root]   (default: current dir)
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

echo "== Python files importing Triton =="
grep -rIl -E 'import[[:space:]]+triton|triton\.language|import[[:space:]]+triton\.language[[:space:]]+as[[:space:]]+tl' \
  --include='*.py' . 2>/dev/null | grep -v -E '/(build|\.git|\.venv|venv|__pycache__|site-packages)/' | sort || echo "(none found)"

echo
echo "== @triton.jit kernels =="
grep -rInE '@triton\.jit|@jit' --include='*.py' . 2>/dev/null \
  | grep -v -E '/(build|\.git|\.venv|venv|__pycache__|site-packages)/' || echo "(none found)"

echo
echo "== Kernel launches (kernel[grid](...)) =="
grep -rInE '\w+\[[^]]*\]\(' --include='*.py' . 2>/dev/null \
  | grep -E 'grid|meta|cdiv|program' | grep -v -E '/(build|\.git|\.venv|venv|__pycache__|site-packages)/' || echo "(none obvious — check launch wrappers)"

echo
echo "== Matmul / XMX candidates (tl.dot) =="
grep -rInE 'tl\.dot' --include='*.py' . 2>/dev/null \
  | grep -v -E '/(build|\.git|\.venv|venv|__pycache__|site-packages)/' || echo "(none)"

echo
echo "== Block pointers / masked memory =="
grep -rInE 'tl\.make_block_ptr|tl\.advance|tl\.load|tl\.store|boundary_check|mask=' \
  --include='*.py' . 2>/dev/null | grep -v -E '/(build|\.git|\.venv|venv|__pycache__|site-packages)/' || echo "(none)"

echo
echo "== Reductions / scans =="
grep -rInE 'tl\.(sum|max|min|cumsum|argmax|argmin)\(' --include='*.py' . 2>/dev/null \
  | grep -v -E '/(build|\.git|\.venv|venv|__pycache__|site-packages)/' || echo "(none)"

echo
echo "== Atomics =="
grep -rInE 'tl\.atomic_(add|max|min|cas|xchg|and|or|xor)' --include='*.py' . 2>/dev/null \
  | grep -v -E '/(build|\.git|\.venv|venv|__pycache__|site-packages)/' || echo "(none)"

echo
echo "== Autotuning / launch config (num_warps / num_stages) =="
grep -rInE '@triton\.autotune|triton\.Config|num_warps|num_stages|tl\.constexpr' \
  --include='*.py' . 2>/dev/null | grep -v -E '/(build|\.git|\.venv|venv|__pycache__|site-packages)/' || echo "(none)"

echo
echo "== PyTorch reference / tests (ready-made correctness oracle) =="
grep -rIlE 'torch\.testing\.assert_close|assert_allclose|@pytest|def test_' \
  --include='*.py' . 2>/dev/null | grep -v -E '/(build|\.git|\.venv|venv|__pycache__|site-packages)/' | head -20 || echo "(none found — a CPU reference must be written)"

echo
echo "Done. Confirm by reading the sources — kernels are often wrapped by autograd.Function or a"
echo "plain Python launcher that builds the grid and passes BLOCK_*/num_warps meta-params."
