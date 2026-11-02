# flashinfer 0.7.2.post1 — gb10 cu134 (0.7.2.post1+gb10.cu134.20261102.0)

Single-arch flashinfer wheels for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FLASHINFER_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261102.0 (UTC date of the tuned commit)
- Built from commit 94327c5cab83b8dcaf68a0820cca2e17214de799; tested at 94327c5cab83b8dcaf68a0820cca2e17214de799
- Upstream base: upstream tag `v0.7.2.post1` (`beabf5004`), merged into `tuned-builds` as `aa8343778`
- sha256_python: fe183d0be5b26eac4552cd78d8e426c04ac80ca68e1c4cd09234a32ffa274e38
- sha256_jit_cache (shim): 760d1de0b6512f64ed631910db67c341998e3e4ac90ff51183c5d7f5593609ef
- sha256_jit_cache_sm121a (provider): c16dfab11d714b682bf2698691ebb14003da728ada82a2768e963f11707b3411

## What changed since 0.7.0+gb10.cu134.20261101.0

- **Upstream merge, 0.7.0 to 0.7.2.post1.** vLLM v0.33.0 imports names that the 0.7.0 fork lacked. Checked by import against this build's venv: the modules `flashinfer.comm.mnnvl_cutedsl_ag`, `flashinfer.comm.mnnvl_cutedsl_moe_finalize`, `flashinfer.comm.tp_moe_finalize_rmsnorm` and `flashinfer.norm.kernels.gated_residual_next_rmsnorm`, and the names `flashinfer.moe_ep.NVLinkOneSidedAlltoAll`, `flashinfer.norm.gated_residual_rmsnorm`, `rmsnorm_sigmoid_gate` and `supports_gated_residual_rmsnorm` now all import. Whether vLLM's code paths that use them work on GB10 has not been run.
- **The 2026-09-18 revert was reverted first.** The fork had reverted an earlier upstream merge (`a08574ef0` reverting `5a31934f0`) because upstream's move of `flashinfer-jit-cache` to provider wheels broke the fork's AOT wheel. A plain merge of the tag gave 636 conflicts. Reverting that revert (`42d3278d5`, 0 conflicts) and then merging the tag left 2 conflicts, in `.gitignore` (kept both sides) and `flashinfer/aot.py` (the `trtllm_utils` AOT condition now covers sm90, sm100, sm103, sm107, sm110, sm120 and sm121).
- **The jit-cache is now three wheels, not two.** Upstream's layout: `flashinfer-jit-cache` is a shim with no shared objects, and the compiled modules are in a provider wheel, `flashinfer-jit-cache-sm121a`, that flashinfer finds through the `flashinfer.jit_cache.providers` entry point. All three carry the same version. Install all three (below). The shim requires the provider at exactly its own version, and the provider is not on PyPI, so pip cannot fetch it from an index: give both URLs.
- **AOT scoping is kept in the provider build.** Upstream's provider backend passed no AOT configuration, which would have dropped the fork's `FLASHINFER_AOT_*` scoping. The provider's `build_backend.py` now reads the same environment variables as the shim did (fa2 head dims 64/128/256, fa3 128 and 192/128, fp16 and bf16; `add_comm`, `add_gemma`, `add_oai_oss` off; `add_moe`, `add_act`, `add_misc`, `add_xqa` on).
- **Size.** The provider has 914 shared objects (913 stamped, 1 host-only) and is 217M. It has **not** been compared with the 0.7.0 jit-cache, so whether the AOT set is larger or smaller than before is not known.

## Tests

Scope (from the attached test log): gates; tuned/tests/jit_cache_coverage.py (JIT disabled, empty workspace). New in this release: a gate that `flashinfer.jit.env.FLASHINFER_AOT_PROVIDERS` finds the provider, and the version pairing check covers all three wheels.
Skips by design: upstream's pytest suite (tests/) is not run (G3, proposed decision pending the owner, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo. **Not run against this release yet:** the vLLM acceptance test, and any Muse Spark or NVLink code path.
Result: === Full test suite summary: gates + jit-cache coverage passed ===

## Install

```
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.2.post1%2Bgb10.cu134.20261102.0/flashinfer_python-0.7.2.post1%2Bgb10.cu134.20261102.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.2.post1%2Bgb10.cu134.20261102.0/flashinfer_jit_cache-0.7.2.post1%2Bgb10.cu134.20261102.0-py3-none-any.whl"
pip install --no-deps "https://github.com/zbrad/flashinfer/releases/download/v0.7.2.post1%2Bgb10.cu134.20261102.0/flashinfer_jit_cache_sm121a-0.7.2.post1%2Bgb10.cu134.20261102.0-cp39-abi3-linux_aarch64.whl"
```
