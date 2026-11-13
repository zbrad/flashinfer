# flashinfer 0.7.4 — gb10 cu134 (0.7.4+gb10.cu134.20261113.1)

Single-arch flashinfer wheels for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FLASHINFER_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261113.1 (UTC date of the tuned commit)
- Built from commit 509371d9288e3ccf7d5d7c170d21f13936d935f4; tested at 509371d9288e3ccf7d5d7c170d21f13936d935f4
- Upstream base: upstream tag `v0.7.4` (`77039051b`), as in 0.7.4+gb10.cu134.20261112.0 (the `upstream/main`
  merge-base, `e5eda939f`, that the generated line named is not the base: the tag is on upstream's release branch)
- Built against torch 2.15.0+gb10.cu134.20261112.0 (`tuned/repo.env`, `ef6600855`); the wheels require only an unpinned
  `torch` and no shared object links a torch library
- sha256_python: ab29bbfe965be6361859d038bfb202558afdbd818d37b991c509439ef14c71ed
- sha256_jit_cache: 7b39885f3fece6ad1e14e33ff7842a40a066223d70e819e28a13807c88b966c4
- sha256_jit_cache_sm121a: faab048cb4b21736dbcb9d2380f2f3dbc4c2fd0f140fee0c318445751d8d7e19

## Changes since 0.7.4+gb10.cu134.20261112.0

- **fp4_quantize on SM 12.1 (`ec0ffc146`).** Upstream maps SM 12.1 to the `fp4_quantization_120f` module whenever
  torch's CUDA is 12.9 or newer; this cache ships `fp4_quantization_121` (sm_121a), so with `FLASHINFER_DISABLE_JIT=1`
  NVFP4 quantization (sglang's NVFP4 path) raised `MissingJITCacheError` on 20261112.0. SM 12.1 now keeps its own
  module; only SM 12.0 is remapped. The native e2m1 conversion is gated on `__CUDA_ARCH__ >= 1000`, so sm_121a has it,
  and the `121` and `120f` modules gave identical quantized values and scale factors on 48 cases on a GB10 node.
- **Test (`069f84adf`).** `jit_cache_coverage.py` gains `check_fp4_quantize` (gb10 only): the public `fp4_quantize`
  with JIT disabled. It fails on the 20261112.0 wheels and passes here.
- **Notes (`509371d92`).** `tuned/docs/gb10_build.md` records the fix.

## Tests

Scope (from the attached test log): gates; tuned/tests/jit_cache_coverage.py (JIT disabled, empty workspace)
Skips by design: upstream's pytest suite (tests/) is not run (G3, proposed decision pending the owner, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo
Result: === Full test suite summary: gates + jit-cache coverage passed ===

## Install

```
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Bgb10.cu134.20261113.1/flashinfer_python-0.7.4%2Bgb10.cu134.20261113.1-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Bgb10.cu134.20261113.1/flashinfer_jit_cache-0.7.4%2Bgb10.cu134.20261113.1-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Bgb10.cu134.20261113.1/flashinfer_jit_cache_sm121a-0.7.4%2Bgb10.cu134.20261113.1-cp39-abi3-linux_aarch64.whl"
```
