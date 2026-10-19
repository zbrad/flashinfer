<!-- DRAFT — not yet published: review every line, then delete this banner and commit. -->
# flashinfer 0.7.0 — gb10 cu134 (0.7.0+gb10.cu134.20261019.0)

Single-arch flashinfer wheels for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FLASHINFER_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261019.0 (UTC date of the tuned commit)
- Built from commit 7feea7de57d3d5bcf30291c094c2a7611939d44a; tested at 7feea7de57d3d5bcf30291c094c2a7611939d44a
- Upstream base: upstream/main merge-base 012542c8523b7e0034fb24a65f0599c6f60b56ff
- sha256_python: 22c6c323b99b90fa98e1b5391e5e0e18832161d3cb5b46e9eab860b01cb28595
- sha256_jit_cache: e75cd25df8461b305e938854b79636cb857abae127dd67fb0368baf20f1ec78f

## Tests

Scope (from the attached test log): gates; tuned/tests/jit_cache_coverage.py (JIT disabled, empty workspace)
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo
Result: === Full test suite summary: gates + jit-cache coverage passed ===

## Install

```
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.0%2Bgb10.cu134.20261019.0/flashinfer_python-0.7.0%2Bgb10.cu134.20261019.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.0%2Bgb10.cu134.20261019.0/flashinfer_jit_cache-0.7.0%2Bgb10.cu134.20261019.0-cp39-abi3-manylinux_2_28_aarch64.whl"
```
