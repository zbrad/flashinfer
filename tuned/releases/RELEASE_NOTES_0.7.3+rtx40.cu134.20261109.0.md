# flashinfer 0.7.3 — rtx40 cu134 (0.7.3+rtx40.cu134.20261109.0)

Single-arch flashinfer wheels for RTX 4080/4090 (Ada Lovelace, SM 89) (FLASHINFER_CUDA_ARCH_LIST=8.9), CUDA 13.4.

- Release label: 20261109.0 (UTC date of the tuned commit)
- Built from commit 03a8d0c7626348099234433c271895553565256e; tested at 7b39423f7859d66c1b09cae950229f1c17595bef
- Upstream base: upstream tag `v0.7.3` (`775207deb`), merged into `tuned-builds` as `a0b08467a`, the same base as
  0.7.3+gb10.cu134.20261106.0. (The tag is on upstream's release branch, so the `upstream/main` merge-base,
  `f4e1f0fcc`, is not the base.)
- torch: built and tested against `2.15.0+rtx40.cu134.20261109.0` (zbrad/pytorch)
- sha256_python: a3c4386d34521321503f5663bb251198571272e667e502a65d27edeb73b4fb14

The first rtx40 release. Only the python wheel ships: rtx40 has no AOT jit-cache (no target-model scope has been
decided for it), so flashinfer's kernels are compiled the first time each one is called.

Built and tested on one RTX 4090 (24 GiB) under WSL2 (Ubuntu 24.04), Windows driver 621.14. Not run on native Linux
or on an RTX 4080.

## Limitations

"Checked" means run against this wheel on that RTX 4090 on 2026-11-09.

- **Kernels compile on first use, and that needs a CUDA toolkit on the machine.** Each kernel variant (op, dtypes,
  head dim) is compiled once, on its first call, into flashinfer's workspace cache (`FLASHINFER_WORKSPACE_BASE`, by
  default under `~/.cache/flashinfer/`); later calls load it. In the test run the modules took 4 to 7 s each to build
  (32 host cores). Checked. The compiler comes from `CUDA_HOME` (or `CUDA_PATH`), then the `nvcc` on `PATH`, then
  `/usr/local/cuda` (`flashinfer/jit/cpp_ext.py`, `get_cuda_path`). With neither variable set and no `nvcc` on `PATH`,
  a decode kernel compiled with this machine's `/usr/local/cuda`, which is CUDA 13.5, and ran. Checked; whether a
  toolkit other than 13.x works was not tested. `ninja` must also be on `PATH`: with it missing, the first call raised
  `FileNotFoundError: ... 'ninja'`. Checked.
- **No AOT jit-cache, so `FLASHINFER_DISABLE_JIT=1` cannot be used** with this release: every kernel would be missing.
  (vLLM's tuned build sets that flag only for gb10.)
- **Single GPU only.** No multi-GPU test was run.

## Tests

Scope (from the attached test log): gates; tuned/tests/jit_kernel_check.py (no jit-cache for this variant: JIT-compiled decode, prefill and paged batch decode vs a float32 reference)
Skips by design: upstream's pytest suite (tests/) is not run (G3, proposed decision pending the owner, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo. **Not run against this release yet:** the vLLM acceptance test (vLLM has no rtx40 pin yet).
Result: === Full test suite summary: gates + jit-cache coverage passed === (fixed text for every variant; for rtx40 the step that ran is jit_kernel_check.py)

**New in this release's test commit (`7b39423f7`):** `tuned/tests/jit_kernel_check.py`, run for any variant without a
jit-cache. Before it, rtx40's full test only checked that flashinfer imports from the wheel; no kernel ran. It
JIT-compiles into an empty workspace and runs single decode (kv 512), causal single prefill (256 x 256) and batch
decode over a paged KV cache (2 requests, page size 16), in bf16 and fp16, with 32 query and 8 KV heads and head_dim
128, and compares each with a float32 torch reference. The tolerances are fixed in the file: max |out - ref| at most
0.02 for bf16 and 0.005 for fp16. Results on this wheel:

| kernel | bf16 | fp16 |
|---|---|---|
| single_decode (kv 512) | 0.00084 | 0.00011 |
| single_prefill causal (256) | 0.00799 | 0.00108 |
| batch_decode paged, request 0 (kv 40) | 0.00193 | 0.00024 |
| batch_decode paged, request 1 (kv 21) | 0.00322 | 0.00049 |

All 6 compiled modules are SM 89 code only, with no PTX (`cuobjdump`). The check can fail: a non-causal prefill
compared with the causal reference gave 3.79.

The same commit fixes `tuned/smoke_test.sh` for variants without a jit-cache: its metadata probe raised on the missing
`flashinfer-jit-cache` package before printing where flashinfer loads from, so the first smoke run failed the
"loads from the venv" check with an empty path. The smoke test of the packaged wheel then passed. The packaging
commit (`22fbc31a7`) changes only `tuned/package.sh`: it copied the python wheel with its build time, so release.sh
always found the test log newer than it. The wheel's bytes and sha256 are unchanged.

## Install

```
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.3%2Brtx40.cu134.20261109.0/flashinfer_python-0.7.3%2Brtx40.cu134.20261109.0-py3-none-any.whl"
```
