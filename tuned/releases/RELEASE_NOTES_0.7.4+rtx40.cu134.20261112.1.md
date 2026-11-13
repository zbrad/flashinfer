# flashinfer 0.7.4 — rtx40 cu134 (0.7.4+rtx40.cu134.20261112.1)

Single-arch flashinfer wheels for RTX 4080/4090 (Ada Lovelace, SM 89) (FLASHINFER_CUDA_ARCH_LIST=8.9), CUDA 13.4.

- Release label: 20261112.1 (UTC date of the tuned commit)
- Built from commit 130552ba5fa2edc5c822a775e6f64275c1b3c6a0; tested at 130552ba5fa2edc5c822a775e6f64275c1b3c6a0
- Upstream base: upstream tag `v0.7.4` (`77039051b`), merged into `tuned-builds` as `e4cfd1d1a`, the same base as
  0.7.4+gb10.cu134.20261112.0. (The tag is on upstream's release branch, so the `upstream/main` merge-base,
  `e5eda939f`, is not the base.)
- torch: built and tested against `2.15.0+rtx40.cu134.20261109.0` (zbrad/pytorch)
- sha256_python: 905d2b9f5c5bb787f868bd0f3a7f4e94fbc8bc39491357993b0a2f5b0895b82b

As with 0.7.3+rtx40, only the python wheel ships: rtx40 has no AOT jit-cache, so flashinfer's kernels are compiled the
first time each one is called, and that needs a CUDA toolkit on the machine. See the 0.7.3+rtx40 notes for details.

Built and tested on one RTX 4090 (24 GiB) under WSL2 (Ubuntu 24.04), Windows driver 621.14. Not run on native Linux
or on an RTX 4080.

## Changes since 0.7.3+rtx40.cu134.20261109.0

- Upstream v0.7.3 -> v0.7.4 (see upstream's release notes).
- `tuned/wheel.sh` refuses to build with untracked files in the packaged trees (`4697f6f5b`) and empties setuptools'
  `build/lib` before the python wheel build (`130552ba5`), so files left from an earlier build cannot end up in the
  wheel.
- No model-level run against this wheel: zbrad/vllm's rtx40 build pins flashinfer 0.7.3+rtx40.

## Tests

Scope (from the attached test log): gates; tuned/tests/jit_kernel_check.py (no jit-cache for this variant: JIT-compiled decode, prefill and paged batch decode vs a float32 reference)
Skips by design: upstream's pytest suite (tests/) is not run (G3, proposed decision pending the owner, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo
Result: === Full test suite summary: gates + jit-cache coverage passed ===

The smoke test of the packaged wheel passed (2026-11-12T20:18:21Z).

## Install

```
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Brtx40.cu134.20261112.1/flashinfer_python-0.7.4%2Brtx40.cu134.20261112.1-py3-none-any.whl"
```
