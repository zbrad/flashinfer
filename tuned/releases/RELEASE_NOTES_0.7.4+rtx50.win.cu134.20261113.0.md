# flashinfer 0.7.4 — rtx50-win cu134 (0.7.4+rtx50.win.cu134.20261113.0) — RTX 50-series, native Windows — pre-release

**Pre-release: a native Windows (MSVC) dev build.** It has had the release checks and a vLLM run on one machine, not
the `tuned/` test suite the Linux releases run (see Tests). It is published so the native Windows vLLM build can pin
it by URL. It is never marked Latest.

Three wheels for RTX 50-series (Blackwell consumer, desktop/laptop, SM 120a) on Windows x86_64
(FLASHINFER_CUDA_ARCH_LIST=12.0a), CUDA 13.4:

- `flashinfer_python`: the Python package (`py3-none-any`).
- `flashinfer_jit_cache_sm120a`: the precompiled kernels, 771 DLLs (`cp39-abi3-win_amd64`; see "Kernels included").
- `flashinfer_jit_cache`: a small package that points flashinfer at the provider wheel above.

Release details:

- Release label: 20261113.0 (UTC date of the build commit; sequence 0, the first label for that date in this repo)
- Built from commit 44e768ddc6efba7f8589350c616cb4d99c870107 (`tuned-win`): the merge of `tuned-builds` at
  `06682123e` into `tuned-win`'s 18 Windows commits at `194f5b74a`. The commit is recorded in each wheel's
  `_build_meta.py`. The build did not record the tree state; the tree was clean at that commit when checked after the
  build.
- Upstream base: upstream tag `v0.7.4` (`77039051b`), merged into `tuned-builds` as `e4cfd1d1a`, the same base as
  0.7.4+rtx40.cu134.20261112.1.
- torch: built against the dev-label build `2.15.0+rtx50.cu134.win.20261109.dev0`, whose repack is the zbrad/pytorch
  pre-release `2.15.0+rtx50.win.cu134.20261109.1`; tested against that pre-release
- apache-tvm-ffi: 0.1.16, pinned (see Install)
- Toolchain: MSVC 14.44.35207 (Windows SDK 10.0.26100.0), CUDA 13.4 (nvcc 13.4.92)
- sha256:
  - `flashinfer_python`: 49dcabe96e32953187872269fce07ee6419d30949f0b146e997a89945267b7bb
  - `flashinfer_jit_cache_sm120a`: a23631ff120e1d91873a130fc50b3921537f6e98766928554ea7d505b32bca9b
  - `flashinfer_jit_cache`: 4af247917504e81885a457133e2ee438f03abc893cc955592b71a9755a5873f3

Built and tested on one RTX 5090 Laptop GPU (24 GiB) with an Intel Core Ultra 9 275HX CPU, Windows 11 Pro
(10.0.26100), driver 621.14, natively (not WSL). It has not been run on a desktop RTX 5090, any other RTX 50-series
card, or another driver.

This is the first published rtx50-win flashinfer. A 0.7.3 build (0.7.3+rtx50.win.cu134.20261112.1) was prepared on
2026-11-12 and not published; this release replaces it.

## Install

In `cmd` (in PowerShell, end the lines with a backtick instead of `^`, and use `$env:FLASHINFER_DISABLE_JIT = '1'`):

```
pip install "torch @ https://github.com/zbrad/pytorch/releases/download/v2.15.0%2Brtx50.win.cu134.20261109.1/torch-2.15.0%2Brtx50.win.cu134.20261109.1-cp314-cp314-win_amd64.whl" ^
  "flashinfer-python @ https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Brtx50.win.cu134.20261113.0/flashinfer_python-0.7.4%2Brtx50.win.cu134.20261113.0-py3-none-any.whl" ^
  "flashinfer-jit-cache @ https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Brtx50.win.cu134.20261113.0/flashinfer_jit_cache-0.7.4%2Brtx50.win.cu134.20261113.0-py3-none-any.whl" ^
  "flashinfer-jit-cache-sm120a @ https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Brtx50.win.cu134.20261113.0/flashinfer_jit_cache_sm120a-0.7.4%2Brtx50.win.cu134.20261113.0-cp39-abi3-win_amd64.whl" ^
  apache-tvm-ffi==0.1.16
set FLASHINFER_DISABLE_JIT=1
```

- **apache-tvm-ffi must be 0.1.16.** The kernel DLLs link `tvm_ffi.dll` from the build environment, which was
  0.1.16. The pin is zbrad/tuned-common's `constraints/rtx50-win.txt`.
- **Set `FLASHINFER_DISABLE_JIT=1`.** All the kernels below then load from the wheel, with no CUDA toolkit or Visual
  Studio on the machine (checked: the release checks ran with neither on `PATH` and `CUDA_PATH` unset). A kernel
  outside the set then raises instead of compiling. Without the flag, flashinfer would try to compile any kernel it
  does not find, and that needs MSVC 14.44, the CUDA 13.4 toolkit and ninja.
- CPython 3.14 only, because the torch wheel is `cp314`.

## Kernels included

771 DLLs. The 770 kernel DLLs have `sm_120a` code only and no PTX (`cuobjdump`, checked on each during the repack);
the 771st, `spdlog`, is host code only:

- **Prefill** (`batch_prefill_with_kv_cache_*`), 72: FA2 prefill. Head dims 64, 128 and 256; fp16 or
  bf16 query; KV cache of the same type, fp8 (e4m3), or fp8 with a per-(token, head) scale; each with and without
  sliding window and logits soft cap (`flashinfer/aot.py`, the two FA2 loops).
- **Decode** (`batch_decode_with_kv_cache_*`), 48: FA2 decode, the same grid without the per-(token, head) scale
  variant (flashinfer reuses the prefill module for that).
- **XQA decode** (`xqa_input_*`), 648: fp16 and bf16 input; KV cache of the same type or fp8 (e4m3); page sizes 16,
  32 and 64; head dims 64, 128 and 256; head group ratios 1 to 8 and 16; with and without sliding window. vLLM uses
  XQA for decode on SM 12x by default.
- `sampling`, `page`, and `spdlog` (host code only, and the one DLL without a build-info stamp).

The same set as the unpublished 0.7.3 build. Not included: MLA, MoE, the fp8/fp4 GEMMs, communication (no NCCL on
Windows), cuDNN and the trtllm kernels. They are not built for Windows yet.

## How the wheels were made from the build

They were built on 2026-11-13 with the dev label `0.7.4+rtx50.cu134.win.20261113.dev0`, then repacked for this release
(the built wheels themselves are unchanged). The repack:

1. **Relabelled them** to the `docs/VERSIONING.md` form (zbrad/tuned-common), `+rtx50.win.cu134.<date>.<seq>`. The
   label is stored only in text files (`_build_meta.py` in each wheel, the provider's `manifest.json`, the shim's
   `_provider_requirements.txt`, and the dist-info) and in the dist-info and `.data` directory names. No DLL
   contains the dev label (checked by searching the bytes of all 771), so the compiled code is the code that was
   built.
2. **Stamped each kernel DLL** with the tuned build info (JSON, schema `tuned-build-info/1`, the fields of the Linux
   wheels' `.tuned_build_info` section) as its Windows version resource. The JSON is the `Comments` string, and
   `FileVersion`/`ProductVersion` are the release label. Read it with `(Get-Item <dll>).VersionInfo`.

## XQA load on Windows

This build includes `194f5b74a` (on `tuned-win`), first made for the unpublished 0.7.3 build. Before it, an XQA module
hung for good at `LoadLibrary` when it was loaded before the process had allocated anything on the GPU. A
namespace-scope static in `csrc/xqa/mha.cu`, `hostSmemSize = configureKernel()`, called `cudaMemcpyFromSymbol` and
`cudaFuncSetAttribute`, and Windows runs a DLL's static initialisers under the loader lock. The fix makes it a
function-local static in `launchMHAFlashInfer`, the form `launchMHA` in the same file already used, so the calls run on
the first launch. The release check loads all XQA modules in a fresh process before any CUDA tensor exists, then runs
one (see Tests).

vLLM was not affected, because it uses CUDA before it loads XQA. `mha_sm90.cu` and `mla_sm120.cu` have the same
pattern, and neither is in this build.

## Limitations

"Checked" means run against these wheels on the machine above on 2026-11-13.

- **Only the kernels listed above.** A model or setting that needs anything else (MLA models such as DeepSeek, MoE,
  fp8/fp4 GEMM, a head dim other than 64/128/256, or a page size other than 16/32/64 for XQA) raises with
  `FLASHINFER_DISABLE_JIT=1`, or needs the toolchain without it.
- **fp8 KV cache:** the FA2 and XQA modules for it are included and the XQA ones load (checked), but no fp8 KV cache
  was run, here or in vLLM.
- **Single GPU only.** There is no NCCL for Windows, and no multi-GPU test was run.
- vLLM printed `[Flashinfer]: Error querying CC state: Not Supported` at startup and ran normally. flashinfer logs
  it when NVML's confidential-computing query (`nvmlSystemGetConfComputeState`) is not supported, as on this laptop
  GPU, and carries on.

## Tests

On the machine above, 2026-11-13. The attached `RELEASE_CHECK_rtx50-win.log` has the output (the Windows user name and
local work paths replaced by placeholders), and `fi_rel_check.py` is the check itself.

- **Release checks (`fi_rel_check.py`), passed:** a new venv with the torch pre-release, the three wheels and their
  PyPI dependencies, `apache-tvm-ffi==0.1.16`, no MSVC or CUDA toolkit on `PATH`, `FLASHINFER_DISABLE_JIT=1`, an empty
  workspace, run from an empty directory. The checks:
  - the three versions;
  - all 648 XQA modules loaded in a fresh process before any CUDA tensor exists, then one run on a 40-token decode
    against a float32 reference: max |out - ref| 2.6e-03;
  - FA2 paged decode at head dims 64/128/256 in bf16 and fp16 against a float32 reference: max 3.8e-03 (bf16) and
    4.7e-04 (fp16);
  - all 648 XQA modules load;
  - causal paged prefill, 40 tokens: 7.7e-03 (bf16) and 9.5e-04 (fp16);
  - top-k/top-p sampling returns only top-k ids;
  - nothing was compiled into the workspace;
  - 770 of the 771 DLLs carry this release's build-info stamp.
  The tolerance is 0.02 for every comparison.
- **vLLM, passed:** vLLM 0.33.1.dev179 (the native Windows dev build from zbrad/vllm `tuned-win`, then named
  `windows-dev`) with these wheels and the torch pre-release, the FLASHINFER attention backend, `Qwen/Qwen3-0.6B`. It
  ran eager and then with `torch.compile` and CUDA graphs, under the same conditions (no toolchain, JIT off). Both runs
  produced greedy completions and a seeded sampled batch, and no DLL was compiled into the workspace.
- **Not run for this build:** upstream's pytest suite. The 0.7.3 dev builds ran `tests/utils/test_sampling.py` and
  `tests/attention/test_xqa.py` on 2026-11-12; they were not rerun on 0.7.4.
- **Not run:** the `tuned/` gates and test suite (bash, which does not run natively on Windows), and any multi-GPU
  test.
