# flashinfer 0.7.3 — gb10 cu134 (0.7.3+gb10.cu134.20261106.0)

Single-arch flashinfer wheels for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FLASHINFER_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261106.0 (UTC date of the tuned commit)
- Built from commit 5bee7ed682228f88ccfe654565b12831e7247ff4; tested at 5bee7ed682228f88ccfe654565b12831e7247ff4
- Upstream base: upstream tag `v0.7.3` (`775207deb`), merged into `tuned-builds` as `a0b08467a`. (The tag is on upstream's release branch, 24 commits not in `upstream/main`, so the `upstream/main` merge-base, `f4e1f0fcc`, is not the base.)
- sha256_python: 95e7755500431c372d29688ace3c6446b6ffd21bc736baca7094a30e0d10a2c6
- sha256_jit_cache (shim): 3cf58666103d9064fee926db26cc7628b42d7836b2e775316471d55fa3e56715
- sha256_jit_cache_sm121a (provider): 04514b3fc9376130a2eb7da0aca829c1ac9b26a26882810abdc7a8f16b6d5516

## What changed since 0.7.2.post1+gb10.cu134.20261102.0

- **Upstream merge, 0.7.2.post1 to 0.7.3** (1,248 upstream commits). 0.7.2.post1 came from upstream's release
  branch and 0.7.3 from main, so git reported 150 conflicts; only 2 were in files the fork changes. `flashinfer/aot.py`
  keeps the fork's SM110/120/121 `trtllm_utils` condition (it already had upstream's new `has_sm107`); everything else
  takes v0.7.3's version, and 12 release-branch files v0.7.3 removed are gone. Checked: the tree equals v0.7.3 except
  the 50 fork-changed files.
- **0.7.2.post1 is still available.** Its release is unchanged, and its source line continues on the
  `tuned-builds-0.7.2` branch. vLLM's `requirements/gb10.txt` still pins 0.7.2.post1.
- **Dependency floors raised by upstream:** `nvidia-cudnn-frontend>=1.32.0` (was 1.31.0) and
  `nvidia-cutlass-dsl>=4.8.0` (was 4.6.2a0), read from the wheel METADATA.
- **The python wheel is smaller.** Upstream's new wheel post-processing (`build_backend._slim_wheel`) drops the
  vendored cutlass/cccl/spdlog files outside their include directories and re-deflates the rest: 133.5 MB to 127.2 MB,
  751 entries dropped (the vendored cutlass and cccl Python packages, tests and examples). flashinfer imports none of
  them; its `cutlass_library` is its own copy under `flashinfer/jit/gemm/cutlass/`. Experimental kernel sources are
  kept (`FLASHINFER_WHEEL_EXCLUDE_EXPERIMENTAL_SOURCES` unset). `tuned/package.sh` ships this wheel as built.
- **The provider has 939 shared objects (938 stamped, 1 host-only), 263 MB** (0.7.2.post1: 914, 217 MB). Compared with
  0.7.2.post1's provider: 26 modules are new, 24 of them FA2 batch-prefill variants for a bf16 query with an fp8
  (e4m3) KV cache and per-token-head scale factors (`token_head_sf`), registered by upstream's AOT list, plus
  `moe_utils` and a re-hashed `cake_sampling`; one old `cake_sampling` hash is gone. The modules in both grew by 3 MB in
  total.

## Tests

Scope (from the attached test log): gates; tuned/tests/jit_cache_coverage.py (JIT disabled, empty workspace)
Skips by design: upstream's pytest suite (tests/) is not run (G3, proposed decision pending the owner, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo. **Not run against this release yet:** the vLLM acceptance test (vLLM still pins 0.7.2.post1).
The smoke test of the packaged wheels also passed (jit_cache_coverage against the package, JIT disabled, empty workspace).
Result: === Full test suite summary: gates + jit-cache coverage passed ===

## Install

```
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.3%2Bgb10.cu134.20261106.0/flashinfer_python-0.7.3%2Bgb10.cu134.20261106.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.3%2Bgb10.cu134.20261106.0/flashinfer_jit_cache-0.7.3%2Bgb10.cu134.20261106.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.3%2Bgb10.cu134.20261106.0/flashinfer_jit_cache_sm121a-0.7.3%2Bgb10.cu134.20261106.0-cp39-abi3-linux_aarch64.whl"
```
