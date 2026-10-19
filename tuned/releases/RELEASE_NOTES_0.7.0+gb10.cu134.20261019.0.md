<!-- DRAFT — not yet published: review every line, then delete this banner and commit. -->
# flashinfer 0.7.0 — gb10 cu134 (0.7.0+gb10.cu134.20261019.0)

Single-arch flashinfer wheels for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FLASHINFER_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261019.0 (UTC date of the tuned commit)
- Built from commit f1f8aaa750b539d09c3f96155e8cb5ba4097753a; tested at e29b6d9eaa0f5e50e7c5d864f38083fe65380c4b
- Upstream base: upstream/main merge-base 012542c8523b7e0034fb24a65f0599c6f60b56ff
- sha256_python: c7383683f582714d217151329aa3062f69d1be030165c659377c4bd9e1a97244
- sha256_jit_cache: d2cd99b6c585c5dc0a8231f97b4555f3541050113c063912f52692ecc93b3963

## Tests

Scope (from the attached test log): gates; tuned/tests/jit_cache_coverage.py (JIT disabled, empty workspace)
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo
Result: === Full test suite summary: gates + jit-cache coverage passed ===

## Install

```
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.0%2Bgb10.cu134.20261019.0/flashinfer_python-0.7.0%2Bgb10.cu134.20261019.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.0%2Bgb10.cu134.20261019.0/flashinfer_jit_cache-0.7.0%2Bgb10.cu134.20261019.0-cp39-abi3-manylinux_2_28_aarch64.whl"
```
