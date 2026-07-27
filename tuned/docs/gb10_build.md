# GB10-Only Build Guide

## Overview

The `tuned-builds` branch (renamed from `native-builds`, itself renamed from `gb10-only`) builds FlashInfer scoped to a single machine: an NVIDIA
GB10 (DGX Spark class, compute capability 12.1 / SM121a). Instead of the full
multi-arch release matrix and the full default AOT kernel matrix, it pins
`FLASHINFER_CUDA_ARCH_LIST=12.1a` and trims the AOT (`flashinfer-jit-cache`)
build to the kernels actually needed by this machine's target models. This
keeps build time down and produces a build that can run on GB10 with **zero
runtime JIT compilation**.

Run it with:

```bash
bash tuned/build.sh gb10
```

The script aborts if `nvidia-smi --query-gpu=compute_cap` isn't `12.1`, so
it's safe to run only on the intended hardware.

## What the script produces

Three things, in order:

1. **Editable install** of `flashinfer-python` into the local dev environment
   (`pip install --no-build-isolation -e .`) — this is what this repo's own
   dev loop and the smoke test use. Kernels still JIT-compile lazily here.
2. **`flashinfer-python` wheel** at `dist/flashinfer_python-*+gb10-*.whl`.
   This wheel carries Python source + `csrc/`/`include/` only — no compiled
   CUDA binary (confirmed by its `py3-none-any` tag). It exists so
   `import flashinfer` works in another venv (e.g. vLLM's).
3. **`flashinfer-jit-cache` wheel** at
   `flashinfer-jit-cache/dist/flashinfer_jit_cache-*+gb10-*.whl`. This is the
   actual GB10 binary: AOT-compiled `.so` files for every kernel variant in
   the filtered config below.

Both wheels are tagged with the `+gb10` local version
(`FLASHINFER_LOCAL_VERSION=gb10`), following the same convention
`.github/workflows/release.yml` uses for CUDA version suffixes (e.g.
`+cu128`).

## Why a separate `flashinfer-jit-cache` wheel is required for "no JIT"

FlashInfer JIT-compiles kernels lazily on first call by default, caching the
`.so` under `~/.cache/flashinfer/`. The plain `flashinfer-python` wheel
**cannot** avoid this — it contains no compiled binary at all.

The only way to get a *no-JIT* runtime is the AOT mechanism:

- `JitSpec.build_and_load()` (`flashinfer/jit/core.py`) checks
  `FLASHINFER_AOT_DIR` for a precompiled `<name>.so` **before** ever touching
  ninja.
- `FLASHINFER_AOT_DIR` resolves to the installed `flashinfer_jit_cache`
  package's bundled directory if that package is present
  (`flashinfer/jit/env.py:_get_aot_dir`).
- Setting `FLASHINFER_DISABLE_JIT=1` turns any AOT cache miss into a hard
  `MissingJITCacheError` instead of a silent JIT compile — use this in any
  environment where you want a guarantee that nothing JIT-compiles.

So shipping a build that "doesn't have to JIT" means: build and install the
`flashinfer-jit-cache` wheel, not just `flashinfer-python`.

## AOT config: scoped to our target models

`flashinfer/aot.py`'s default AOT config compiles the full kernel matrix
(~3800 build steps for a single-arch build) — every dtype, head_dim, and
optional kernel family. `tuned/build.sh gb10` instead sets `FLASHINFER_AOT_*`
env vars (read by `flashinfer-jit-cache/build_backend.py`'s
`_config_overrides_from_env()`, which mirrors the CLI flags in
`flashinfer/aot.py`'s `main()`) to scope the build to our two target models:

- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` — NemotronH hybrid
  Mamba+Attention, head_dim=128 GQA, bf16, NVFP4 MoE (512 routed experts).
- `deepseek-ai/DeepSeek-V4-Flash` — MLA attention (head_dim=512,
  qk_rope_head_dim=64), bf16, fp8 block-scaled MoE (256 routed experts).

| `FLASHINFER_AOT_*` env var | Value | Why |
|---|---|---|
| `FLASHINFER_AOT_FA2_HEAD_DIM` | `128,128` | Both models' dense/GQA attention uses head_dim=128 |
| `FLASHINFER_AOT_FA3_HEAD_DIM` | `128,128 192,128` | `192,128` kept as a safety margin in case DeepSeek-V4's generic FA3 path (prefill fallback, MTP, etc.) ever needs the legacy MLA shape |
| `FLASHINFER_AOT_F16_DTYPE` | `bfloat16` | Both models are bf16-base; drops fp16 |
| `FLASHINFER_AOT_ADD_COMM` | `false` | GB10 is a single GPU; no NCCL-style collective kernels needed |
| `FLASHINFER_AOT_ADD_GEMMA` | `false` | Neither target model is Gemma |
| `FLASHINFER_AOT_ADD_OAI_OSS` | `false` | Neither target model is gpt-oss |
| `FLASHINFER_AOT_ADD_MOE` | `true` | Both targets are MoE (fp4/fp8) |
| `FLASHINFER_AOT_ADD_ACT` | `true` | Activation kernels (small, kept) |
| `FLASHINFER_AOT_ADD_MISC` | `true` | Includes norm/rope/sampling/page **and Mamba SSU** (see below) — required for Nemotron-3-Super |
| `FLASHINFER_AOT_ADD_XQA` | `false` | Not used by either target model on this backend |

`f8_dtype` is left at the flashinfer default (`float8_e4m3fn`), which matches
both models' fp8 paths.

This cuts the AOT build from **3807 ops to 111**.

DeepSeek-V4's MLA path is served by `gen_sparse_mla_sm120_module()`, which is
unconditional for SM120/SM121 in `flashinfer/aot.py` (not gated by any
`add_*` flag, explicitly comment-labeled "DSv4 + DSv3.2 / GLM5.1") — no
config flag is needed for it.

To rescope for a different model, see `flashinfer/aot.py`'s `main()` for the
full list of CLI-flag-equivalent env vars, or read
`get_default_config()`/`gen_all_modules()` to find which kernel family a
given architecture needs.

## Bug fixed: Mamba SSU was never AOT-cacheable on Blackwell

While scoping the build for Nemotron-3-Super (a hybrid Mamba+attention
model), we found `flashinfer/aot.py`'s `gen_all_modules()` never registered
`gen_selective_state_update_sm100_module` — but the runtime dispatcher
(`flashinfer/mamba/selective_state_update.py`) always picks that exact module
for any GPU with `sm_major >= 10` (SM100/110/120/121 — all of Blackwell, not
GB10-specific). Without the fix, **no** AOT config could ever serve the Mamba
SSU op on this hardware family; it would silently JIT or hard-fail under
`FLASHINFER_DISABLE_JIT=1`.

Fixed locally in this branch (commit `977e70a9`) and reported upstream:
[flashinfer-ai/flashinfer#3775](https://github.com/flashinfer-ai/flashinfer/issues/3775).
If `aot.py` is ever rebased from upstream `main`, re-verify this fix survived
the merge before assuming Mamba/SSU is covered.

## Installing into vLLM

```bash
<vllm-venv>/bin/pip install --force-reinstall \
  <flashinfer-repo>/dist/flashinfer_python-*+gb10-*.whl \
  <flashinfer-repo>/flashinfer-jit-cache/dist/flashinfer_jit_cache-*+gb10-*.whl
```

Then, in vLLM's environment:

```bash
export FLASHINFER_DISABLE_JIT=1   # hard-fail instead of silently JIT-compiling on a cache miss
```

### vLLM-side flags needed to actually exercise these kernels

Installing the wheels is necessary but not sufficient — vLLM has to be told
to route through FlashInfer for the relevant ops, or it'll use its own
defaults and never touch the AOT cache at all:

- **Mamba SSU (Nemotron-3-Super)**: vLLM defaults to its own Triton SSU
  kernel (`vllm/model_executor/layers/mamba/ops/ssu_dispatch.py`,
  `MambaBackendEnum.TRITON`). Pass `--mamba-backend flashinfer` (or set
  `MambaConfig.backend = "flashinfer"`) to route through
  `flashinfer.mamba.selective_state_update` — only then does the SSU AOT fix
  above actually get exercised. Requires `flashinfer-python >= 0.6.4` per
  vLLM's own check.
- **Attention/MoE backends**: confirm whatever attention/MoE backend
  selection vLLM is configured with on this deployment actually resolves to
  FlashInfer's kernels (vLLM has multiple backend options per op family);
  this doc only covers what we've verified (standard decode attention and
  Mamba SSU loading from the AOT cache without JIT) — re-verify any other op
  path you rely on the same way before assuming it's covered by the 111-op
  filtered build.

### Verifying no-JIT end to end

```python
import torch, flashinfer
# FLASHINFER_DISABLE_JIT=1 must be set in the environment before this runs.

# Standard attention (Nemotron-3-Super's GQA path)
q = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(16, 8, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(16, 8, 128, dtype=torch.bfloat16, device="cuda")
flashinfer.single_decode_with_kv_cache(q, k, v)  # must NOT raise MissingJITCacheError
```

A `flashinfer.jit.core.MissingJITCacheError` here means some op your
deployment needs isn't in the filtered AOT config — add it back via the
`FLASHINFER_AOT_*` env vars in `tuned/build.sh` and rebuild.
