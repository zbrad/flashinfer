# flashinfer 0.7.0 — gb10 cu134 (0.7.0+gb10.cu134.20261101.0)

Single-arch flashinfer wheels for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FLASHINFER_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261101.0 (UTC date of the tuned commit)
- Built from commit 30db3d6444193e8619a57f26bbd0d3ac07a16c82; tested at 30db3d6444193e8619a57f26bbd0d3ac07a16c82
- Upstream base: upstream/main merge-base 012542c8523b7e0034fb24a65f0599c6f60b56ff
- sha256_python: 56aed5cf05680bcc42cb3cd5ab062c9f1cb322bcef4b4e922481f2f2a70c0632
- sha256_jit_cache: 3a7b782e3624d957102220cb73839b982fc1cbe13eb5c77ab74577af781e58d4

## Changes since 0.7.0+gb10.cu134.20261030.0

- Built against torch 2.15.0+gb10.cu134.20261030.1, where flashinfer 20261030.0 was built against torch 20261030.0. The only difference
  in that torch is its source: upstream's v2.15.0 tag instead of the earlier base. Its build configuration, library names
  and JSON build info are the same.
- No kernel or source changes: the only file outside the release notes that changed since 20261030.0's build commit
  (c8f23af53) is `tuned/repo.env`, which pins the torch wheel.
- Not tested beyond this release's gates: the model-level check (vllm's acceptance test) has not been run against these
  wheels.

## Tests

Scope (from the attached test log): gates; tuned/tests/jit_cache_coverage.py (JIT disabled, empty workspace)
Skips by design: upstream's pytest suite (tests/) is not run (G3, proposed decision pending the owner, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo
Result: === Full test suite summary: gates + jit-cache coverage passed ===

## Install

```
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.0%2Bgb10.cu134.20261101.0/flashinfer_python-0.7.0%2Bgb10.cu134.20261101.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.0%2Bgb10.cu134.20261101.0/flashinfer_jit_cache-0.7.0%2Bgb10.cu134.20261101.0-cp39-abi3-manylinux_2_28_aarch64.whl"
```
