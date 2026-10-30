# flashinfer 0.7.0 — gb10 cu134 (0.7.0+gb10.cu134.20261030.0)

Single-arch flashinfer wheels for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FLASHINFER_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261030.0 (UTC date of the tuned commit)
- Built from commit c8f23af53f3f40c47edbef8679b4b2a68041a596; tested at c8f23af53f3f40c47edbef8679b4b2a68041a596
- Upstream base: upstream/main merge-base 012542c8523b7e0034fb24a65f0599c6f60b56ff
- sha256_python: faccc05e7c8b15d495056edbe0e57541a3ed9e8852f7c5fa5c5a68aeb75bb594
- sha256_jit_cache: 8aafd2a0cb287f6e2abb2a0bd39438e08b8fd962c6992c8418226927a15acd23

## Changes since 0.7.0+gb10.cu134.20261028.0

- Built against torch 2.15.0+gb10.cu134.20261030.0, where 20261028.0 was built against 20261022.0. It is the same torch
  source; 20261030.0 carries JSON build info (`.tuned_build_info`) instead of the free-text `.pytorch_build_info` stamp, and
  `tuned/build.sh` now checks the installed torch with `gpu_tuned_verify_build_info_json`.
- The vendored tuned-common is synced to fc2a630: 9a61eb4's venv tool-resolution check, toolkit PATH gate and locked
  build requirements, and fc2a630's `gpu_tuned_assert_versioned_runpath`.
- No kernel or source changes: nothing outside `tuned/` changed since 20261028.0's build commit (b3a714164).

## Tests

Scope (from the attached test log): gates; tuned/tests/jit_cache_coverage.py (JIT disabled, empty workspace)
Skips by design: upstream's pytest suite (tests/) is not run (G3, proposed decision pending the owner, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo
Result: === Full test suite summary: gates + jit-cache coverage passed ===

## Install

```
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.0%2Bgb10.cu134.20261030.0/flashinfer_python-0.7.0%2Bgb10.cu134.20261030.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.0%2Bgb10.cu134.20261030.0/flashinfer_jit_cache-0.7.0%2Bgb10.cu134.20261030.0-cp39-abi3-manylinux_2_28_aarch64.whl"
```
