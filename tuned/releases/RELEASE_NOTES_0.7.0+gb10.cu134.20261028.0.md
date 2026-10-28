# flashinfer 0.7.0 — gb10 cu134 (0.7.0+gb10.cu134.20261028.0)

Single-arch flashinfer wheels for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FLASHINFER_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261028.0 (UTC date of the tuned commit)
- Built from commit b3a714164ea04025d82cadb9b8d49f658d680db4; tested at b3a714164ea04025d82cadb9b8d49f658d680db4
- Upstream base: upstream/main merge-base 012542c8523b7e0034fb24a65f0599c6f60b56ff
- sha256_python: 5598f4042fd57f152c1f22add9e4af889c228602231e51f01f4d743672fe29a0
- sha256_jit_cache: d287ab40c31b6af9384d9aae2be0899984cb4436e4debf10d3b6445ecbf3ca9e

## Changes since 0.7.0+gb10.cu134.20261019.0

- Built against torch 2.15.0+gb10.cu134.20261022.0 (no CPU BLAS, LAPACK or MPI), where 20261019.0 was built against 20261012, which links OpenMPI.
- The build venv has setuptools 84.2 or newer, and package.sh now checks that and audits each wheel's runtime dependencies (no vulnerable package with a fix available).
- No kernel or source changes from this fork's tuned code: this release rebuilds the same upstream base with the new torch and build gates.

## Tests

Scope (from the attached test log): gates; tuned/tests/jit_cache_coverage.py (JIT disabled, empty workspace)
Skips by design: upstream's pytest suite (tests/) is not run (G3, proposed decision pending the owner, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo
Result: === Full test suite summary: gates + jit-cache coverage passed ===

## Install

```
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.0%2Bgb10.cu134.20261028.0/flashinfer_python-0.7.0%2Bgb10.cu134.20261028.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.0%2Bgb10.cu134.20261028.0/flashinfer_jit_cache-0.7.0%2Bgb10.cu134.20261028.0-cp39-abi3-manylinux_2_28_aarch64.whl"
```
