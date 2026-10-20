# flashinfer 0.7.0 — gb10 cu134 (0.7.0+gb10.cu134.20261019.0)

Single-arch flashinfer wheels for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FLASHINFER_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261019.0 (UTC date of the tuned commit)
- Built from commit 7feea7de57d3d5bcf30291c094c2a7611939d44a; tested at 7feea7de57d3d5bcf30291c094c2a7611939d44a
- Upstream base: upstream/main merge-base 012542c8523b7e0034fb24a65f0599c6f60b56ff
- sha256_python: 22c6c323b99b90fa98e1b5391e5e0e18832161d3cb5b46e9eab860b01cb28595
- sha256_jit_cache: 4031875f4f7894d1159f5510f48d6a727660f25afa8511439843b175afb0bead

## What changed in this release

- **The jit-cache now holds the XQA and `trtllm_utils` kernels.** The previous GB10 jit-cache (`0.7.0+gb10.cu134.tuning.35`) held 146 precompiled objects
  and had no `xqa` and no `trtllm_utils` module (checked 2026-10-19 while testing vLLM 0.32 with the NVFP4 NemotronH stand-in). That stand-in then
  failed with `MissingJITCacheError` under `FLASHINFER_DISABLE_JIT=1`, and compiled both modules at runtime when JIT was allowed. This cache holds 174 objects,
  27 of them `xqa*` (24 not MLA) and one `trtllm_utils.so`.
- **Two changes in `flashinfer/aot.py`:** `trtllm_utils` is registered for SM 11.0, 12.0 and 12.1 as well as 9.0 and 10.0 (the runtime builds it for any SM12x);
  and the XQA shape lists can be set with `FLASHINFER_AOT_XQA_{INPUT_TYPE,FP8_KV_CACHE,TOKEN_PER_PAGE,HEAD_SIZE,HEAD_GRP_SIZE}`, which
  `tuned/repo.env` sets to bfloat16, fp8 e4m3 KV on and off, pages 16/32/64, head_dim 128, head group ratio 8 and 16. Upstream's defaults omit group ratio 16, which
  the NemotronH stand-in needs.
- **Release process moved to the tuned-builds architecture** (docs/ARCHITECTURE.md): the wheels are named by date label instead of the retired `tuning.N` counter,
  carry JSON build info on every object with device code (173 of 174; one is host-only), and `package.sh` fails if any object holds another architecture or any PTX.
  Builds no longer install into `~/.local`.

## Tests

Scope (from the attached test log): gates; tuned/tests/jit_cache_coverage.py (JIT disabled, empty workspace)
Skips by design: upstream's pytest suite (tests/) is not run (G3, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo
Result: === Full test suite summary: gates + jit-cache coverage passed ===

- **Gates (full_test.sh log):** 174 jit-cache objects checked; the 173 with device code are exactly `sm_121a` with no PTX, and the toolkit libraries resolve under `/usr/local/cuda-13.4`.
- **Coverage test (`tuned/tests/jit_cache_coverage.py`, `FLASHINFER_DISABLE_JIT=1`, empty workspace):** `trtllm_utils`, all 12 configured XQA shapes and a
  `BatchDecodeWithPagedKVCacheWrapper` run (bf16, head_dim 128) loaded from the cache; no shared object was compiled into the workspace.
- **Smoke test (attached log):** the packaged wheels install, their sha256 and sidecars match, all 174 installed objects are byte-identical to the packaged
  copies, 173 stamps read back as this release, and the coverage test passes against the package.

## Additional tests before release

Run on the packaged wheels with `CUDA_TAG=cu134 bash tuned/tests/extra_tests.sh gb10 acceptance|flashinfer` (committed; not part of the gates).
Run 2026-10-19 on one GB10.

- **vLLM 0.32 acceptance test** (`tuned/gb10_acceptance_test.sh` in zbrad/vllm, these two wheels in its build venv, `FLASHINFER_DISABLE_JIT=1`, empty
  `FLASHINFER_WORKSPACE_BASE`, `ACCEPT_GPU_MEM_UTIL=0.25`): `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` each loaded and
  generated text, and no shared object was compiled into the workspace. Four stand-ins were skipped by that script (DeepSeek-V4 tiny and Pro-Tiny, Qwen2-VL tiny,
  DeepSeek-V3 tiny), so **DeepSeek-V4 and Qwen2-VL are untested with this release**. With the previous cache (`tuning.35`) the 30B NVFP4 stand-in failed this same check (2026-10-19, `MissingJITCacheError` in `get_xqa_module`).
- **flashinfer's own tests, XQA batch decode** (`tests/attention/test_xqa_batch_decode.py::test_xqa_batch_decode`, the 288 ids for q bf16, kv bf16 or fp8,
  o bf16, pages 16/32/64, head group 8 and 16, head_dim 128): 192 passed, 96 skipped by the test itself ("Mask is unused for q_len_per_req == 1"), JIT allowed. Four
  modules were compiled at run time because they are not in the cache: the speculative-decode variants (`use_spec_dec_True`, `spec_q_seq_len_4`, page size 32, group
  ratio 16; with and without sliding window; kv bf16 and e4m3). No other module was compiled into the empty workspace, so the remaining selected ids ran from the cache.
- **flashinfer's own tests, Mamba selective state update** (`tests/mamba/test_selective_state_update_stp.py`, all 83, run in vLLM's venv for triton): 83 passed in 188 s,
  JIT allowed. 14 modules were compiled at run time, test-only variants (other state dtypes and state sizes); they were not compared with the cache.

## Known limits

- **vLLM's pin is not bumped yet.** The acceptance test above ran with these wheels installed by hand into vLLM's build venv. vLLM's `requirements/gb10.txt` still pins
  `0.7.0+gb10.cu134.tuning.35` until it is bumped to this release and vLLM's own `full_test` is re-run.
- **Speculative decoding and sliding-window XQA are not precompiled** (see the XQA test above); they compile at first use, or fail under `FLASHINFER_DISABLE_JIT=1`.
- **Only the listed shapes are precompiled.** Anything outside the lists in `tuned/repo.env` still needs JIT, or fails with `MissingJITCacheError` under
  `FLASHINFER_DISABLE_JIT=1`. For example `single_decode_with_kv_cache` is not in the AOT set (batch decode is); a call to it compiles at first use.
- **Narrow test gate.** Upstream's pytest suite (`tests/`) and multi-GPU tests are not run. This narrower gate is a proposal; the owner has not yet confirmed it as a
  recorded decision (the test log header calls it one, which predates that), and no mapped-library check of a running process exists (G2).
- **Older releases stay.** The earlier `tuning.N` releases are not renamed or removed; consumers that pin them keep working.
- **The install commands below were not run against the published URLs** (the release is not published yet). The wheels were installed into a build venv from the local files.

## Install

```
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.0%2Bgb10.cu134.20261019.0/flashinfer_python-0.7.0%2Bgb10.cu134.20261019.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.0%2Bgb10.cu134.20261019.0/flashinfer_jit_cache-0.7.0%2Bgb10.cu134.20261019.0-cp39-abi3-manylinux_2_28_aarch64.whl"
```
