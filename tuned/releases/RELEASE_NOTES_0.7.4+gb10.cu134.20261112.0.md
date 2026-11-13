# flashinfer 0.7.4 — gb10 cu134 (0.7.4+gb10.cu134.20261112.0)

Single-arch flashinfer wheels for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FLASHINFER_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261112.0 (UTC date of the tuned commit)
- Built from commit e4cfd1d1a3e47a736e754b2f181fc0515de1182f; tested at 34aca6bca3c1733d54b9eb25b6932eeb59acf434
- Upstream base: upstream tag `v0.7.4` (`77039051b`), merged into `tuned-builds` as `e4cfd1d1a`. (The tag is on upstream's release branch, 20 commits not in `upstream/main`, so the `upstream/main` merge-base, `e5eda939f`, is not the base. Corrected after publishing; the release tag points at the notes as first published, which named that merge-base.)
- sha256_python: 238b3bd2467c37c30838ff1c3dcd8faa50b15c5e95f794a74fbce6fc8d7c990e
- sha256_jit_cache: 381d6f587883bca1915e4a5a280a4304aa898b279eab4298c92387663f7a6c6f
- sha256_jit_cache_sm121a: fcf63833b0be954a19dee4261854f0774fe2a471a69f87e5185e9b0fda4dfc87

## Tests

Scope (from the attached test log): gates; tuned/tests/jit_cache_coverage.py (JIT disabled, empty workspace)
Skips by design: upstream's pytest suite (tests/) is not run (G3, proposed decision pending the owner, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo
Result: === Full test suite summary: gates + jit-cache coverage passed ===

0.7.3 -> 0.7.4 adds upstream's `cake_vsa_sm120` modules (943 jit-cache modules vs 939). No model-level run against
0.7.4 yet: zbrad/vllm still pins 0.7.2.post1, and zbrad/sglang's acceptance test on this release (in progress) is the
first. The PTX gate was corrected before this release: it now reads only cuobjdump's `PTX file <n>:` entries (34aca6bca);
the old pattern matched `sm_121a` in the cake_vsa file names.

## Install

```
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Bgb10.cu134.20261112.0/flashinfer_python-0.7.4%2Bgb10.cu134.20261112.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Bgb10.cu134.20261112.0/flashinfer_jit_cache-0.7.4%2Bgb10.cu134.20261112.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Bgb10.cu134.20261112.0/flashinfer_jit_cache_sm121a-0.7.4%2Bgb10.cu134.20261112.0-cp39-abi3-linux_aarch64.whl"
```
