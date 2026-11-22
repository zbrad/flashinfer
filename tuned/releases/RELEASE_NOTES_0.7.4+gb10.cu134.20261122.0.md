# flashinfer 0.7.4 — gb10 cu134 (0.7.4+gb10.cu134.20261122.0)

Single-arch flashinfer wheels for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FLASHINFER_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261122.0 (UTC date of the tuned commit)
- Built from commit 0664c90185b19381d32f9d5f28568343a43942e0; tested at 0664c90185b19381d32f9d5f28568343a43942e0
- Upstream base: upstream tag `v0.7.4` (`77039051b`), as in 0.7.4+gb10.cu134.20261113.1 (the `upstream/main`
  merge-base, `e5eda939f`, that the generated line named is not the base: the tag is on upstream's release branch)
- Built against torch 2.15.0+gb10.cu134.20261112.0 (`tuned/repo.env`, unchanged); the wheels require only an unpinned
  `torch` and no shared object links a torch library
- sha256_python: 443d6f5fa598f62bb7708153848ff0ec3b9002d14fa9ca58a0962c6efb1769f9
- sha256_jit_cache: 55084420d5fdbb6b1b069d733c2f46ef9cf2b82be1dee7c3f73a1ad2e3e6f031
- sha256_jit_cache_sm121a: f7e2dcc9f3cbd2164a2f3cdf6ac12b6e08e2d092c1ad4a04b39a4bc0fb4b3c4a

## Changes since 0.7.4+gb10.cu134.20261113.1

- **No run-time download from the DSL FMHA path with `FLASHINFER_NO_DOWNLOAD` set (`c2ad09046`).**
  `attention/cute_dsl/fmha.py:_get_checksums()` fetched `checksums.txt` from NVIDIA's artifact repository with
  `download_file()` directly whenever the local copy was missing; it was the one run-time download that
  `FLASHINFER_NO_DOWNLOAD` did not stop (`jit/cubin_loader.get_artifact()` checks it). It now raises naming the
  missing file; a copy already on disk is still used. zbrad/vllm tuned-builds defaults `FLASHINFER_NO_DOWNLOAD=1`
  in its gb10 wheels from the next release (`2553af3155`), so FlashInfer there runs installed kernels only.
  `tests/test_artifacts.py` gains a test that fails on the old code.
- **tuned-common syncs** (`f6b6904f4` 6450ccb, `8d48848cf` addd9fe, `0664c9018` 7f8a05d): build-script changes only;
  7f8a05d adds SemVer native-tag functions that wheel repos do not use.
- The AOT kernel set (`tuned/repo.env`) and every other source file are unchanged since 20261113.1; the jit-cache is
  rebuilt from the same configuration.

## Tests

Scope (from the attached test log): gates; tuned/tests/jit_cache_coverage.py (JIT disabled, empty workspace)
Skips by design: upstream's pytest suite (tests/) is not run (G3, proposed decision pending the owner, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo
Result: === Full test suite summary: gates + jit-cache coverage passed ===

Before release (`tuned/tests/extra_tests.sh gb10 acceptance`: these packaged wheels in the vllm 0.34.1.dev214 venv,
`FLASHINFER_DISABLE_JIT=1`, empty `FLASHINFER_WORKSPACE_BASE`): 4 PASS (Nemotron-3-Nano-4B-BF16,
Nemotron-3-Nano-30B-A3B-NVFP4, deepseek-v4-tiny-random, qwen2-vl-tiny-random), 3 documented SKIPs (bad-checkpoint,
mla-dims-too-small, trtllm-moe-sm100-only); 0 shared objects compiled into the workspace.

## Install

```
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Bgb10.cu134.20261122.0/flashinfer_python-0.7.4%2Bgb10.cu134.20261122.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Bgb10.cu134.20261122.0/flashinfer_jit_cache-0.7.4%2Bgb10.cu134.20261122.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Bgb10.cu134.20261122.0/flashinfer_jit_cache_sm121a-0.7.4%2Bgb10.cu134.20261122.0-cp39-abi3-linux_aarch64.whl"
```
