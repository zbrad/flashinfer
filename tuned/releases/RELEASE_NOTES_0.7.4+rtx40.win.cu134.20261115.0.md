# flashinfer 0.7.4 — rtx40-win cu134 (0.7.4+rtx40.win.cu134.20261115.0) — RTX 40-series, native Windows — pre-release

**Pre-release: a native Windows (MSVC) dev build.** It has had the release checks on one machine, not the `tuned/`
test suite the Linux releases run, and no vLLM run yet (see Tests). It is published so the native Windows vLLM build
can pin it by URL. It is never marked Latest.

Three wheels for RTX 4080/4090 (Ada Lovelace, SM 89) on Windows x86_64 (FLASHINFER_CUDA_ARCH_LIST=8.9), CUDA 13.4:

- `flashinfer_python`: the Python package (`py3-none-any`).
- `flashinfer_jit_cache_sm89`: the precompiled kernels, 123 DLLs (`cp39-abi3-win_amd64`; see "Kernels included").
- `flashinfer_jit_cache`: a small package that points flashinfer at the provider wheel above.

Release details:

- Release label: 20261115.0 (UTC date of the build commit; sequence 0, the first label for that date in this repo)
- Built from commit e2658a2228b (`tuned-win`), the commit 0.7.4+rtx50.win.cu134.20261113.0 was built from
  (`44e768ddc`) plus that release's notes; no source change. The commit is recorded in each wheel's `_build_meta.py`;
  the tree was clean at the end of the build.
- Upstream base: upstream tag `v0.7.4` (`77039051b`), the same base as the rtx50-win and Linux rtx40 0.7.4 wheels.
- torch: built and tested against the zbrad/pytorch pre-release `2.15.0+rtx40.win.cu134.20261115.1`
- apache-tvm-ffi: 0.1.16, pinned (see Install)
- Toolchain: MSVC 14.42.34433 (Windows SDK 10.0.26100.0), CUDA 13.4 (nvcc 13.4.59)
- sha256:
  - `flashinfer_python`: c3e9336300f9ae53512048b802cc8d73e6e9b54ded52d64a6b6802473e7589ed
  - `flashinfer_jit_cache_sm89`: b2d1201a1c7d2826682466d70011d8753eff015929488466102fffea1aca6aef
  - `flashinfer_jit_cache`: 0c2ddca1f30dc8d8009eb9d18151b55613370269ce1cca0b23113c1a6f700385

Built and tested on one desktop RTX 4090 (24 GiB) with an Intel Core i9-14900KF CPU, Windows 11 Pro (10.0.26200),
driver 621.14, natively (not WSL). It has not been run on an RTX 4080, a laptop RTX 40, or another driver.

## Install

In `cmd` (in PowerShell, end the lines with a backtick instead of `^`, and use `$env:FLASHINFER_DISABLE_JIT = '1'`):

```
pip install "torch @ https://github.com/zbrad/pytorch/releases/download/v2.15.0%2Brtx40.win.cu134.20261115.1/torch-2.15.0%2Brtx40.win.cu134.20261115.1-cp314-cp314-win_amd64.whl" ^
  "flashinfer-python @ https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Brtx40.win.cu134.20261115.0/flashinfer_python-0.7.4%2Brtx40.win.cu134.20261115.0-py3-none-any.whl" ^
  "flashinfer-jit-cache @ https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Brtx40.win.cu134.20261115.0/flashinfer_jit_cache-0.7.4%2Brtx40.win.cu134.20261115.0-py3-none-any.whl" ^
  "flashinfer-jit-cache-sm89 @ https://github.com/zbrad/flashinfer/releases/download/v0.7.4%2Brtx40.win.cu134.20261115.0/flashinfer_jit_cache_sm89-0.7.4%2Brtx40.win.cu134.20261115.0-cp39-abi3-win_amd64.whl" ^
  apache-tvm-ffi==0.1.16
set FLASHINFER_DISABLE_JIT=1
```

- **apache-tvm-ffi must be 0.1.16.** The kernel DLLs link `tvm_ffi.dll` from the build environment, which was
  0.1.16. The pin is zbrad/tuned-common's `constraints/rtx40-win.txt`.
- **Set `FLASHINFER_DISABLE_JIT=1`.** All the kernels below then load from the wheel, with no CUDA toolkit or Visual
  Studio on the machine (checked: the release checks ran with neither on `PATH` and `CUDA_PATH` unset). A kernel
  outside the set then raises instead of compiling. Without the flag, flashinfer would try to compile any kernel it
  does not find, and that needs MSVC, the CUDA 13.4 toolkit and ninja.
- CPython 3.14 only, because the torch wheel is `cp314`.

## Kernels included

123 DLLs. The 122 kernel DLLs have `sm_89` code only and no PTX (`cuobjdump`, checked on each during packaging); the
123rd, `spdlog`, is host code only. The dense-model set of the rtx50-win wheels, without XQA:

- **Prefill** (`batch_prefill_with_kv_cache_*`), 72: FA2 prefill. Head dims 64, 128 and 256; fp16 or bf16 query; KV
  cache of the same type, fp8 (e4m3), or fp8 with a per-(token, head) scale; each with and without sliding window and
  logits soft cap (`flashinfer/aot.py`, the two FA2 loops, with the same AOT settings as rtx50-win).
- **Decode** (`batch_decode_with_kv_cache_*`), 48: FA2 decode, the same grid without the per-(token, head) scale
  variant.
- `sampling`, `page`, and `spdlog` (host code only, and the one DLL without a build-info stamp).

**No XQA:** flashinfer builds XQA only for SM 9.x, 10.x and 12.x (`flashinfer/jit/xqa.py`,
`supported_major_versions=[9, 10, 12]`), so there is none for SM 89. Also not included, as for rtx50-win: MLA, MoE,
the fp8/fp4 GEMMs, communication (no NCCL on Windows), cuDNN and the trtllm kernels.

## How the wheels were made

1. **Built with the release label** (`FLASHINFER_LOCAL_VERSION=rtx40.win.cu134.20261115.0`): all three wheels at
   once, from one commit. There is no relabel step (the rtx50-win wheels were built with a dev label and relabelled).
2. **Build paths mapped at compile time**, as in the torch rtx40-win and rtx50-win wheels: the `CL` environment
   variable carried `/experimental:deterministic` and `/pathmap` for the source checkout (which also holds the AOT
   build directory), the build venv and the Python installation. cl reads `CL` on every invocation, including the ones
   nvcc makes. This is the first flashinfer Windows release built this way; the rtx50-win 0.7.4 wheels were not.
3. **Stamped each kernel DLL** with the tuned build info (JSON, schema `tuned-build-info/1`) as its Windows version
   resource. The JSON is the `Comments` string, and `FileVersion`/`ProductVersion` are the release label. Read it with
   `(Get-Item <dll>).VersionInfo`. `flashinfer_python` and `flashinfer_jit_cache` are published as built.
4. **A private-path gate** searched every file in the three wheels, as written and with NUL bytes removed, for
   absolute paths under the build machine's source, build and user-profile directories and for the machine's host and
   user names. 0 files matched.

## Limitations

"Checked" means run against these wheels on the machine above on 2026-11-18 (UTC).

- **Only the kernels listed above.** A model or setting that needs anything else (MLA models such as DeepSeek, MoE,
  fp8/fp4 GEMM, a head dim other than 64/128/256) raises with `FLASHINFER_DISABLE_JIT=1`, or needs the toolchain
  without it.
- **No vLLM run yet.** There is no rtx40-win vLLM build yet; which of these kernels vLLM's FLASHINFER backend asks for
  on SM 89 has not been observed.
- **fp8 KV cache:** the FA2 modules for it are included, but none was run.
- **Single GPU only.** There is no NCCL for Windows, and no multi-GPU test was run.
- With JIT disabled, flashinfer still writes each looked-up module's generated sources (`.cu`, `.inc`) and an empty
  `flashinfer_jit.log` into its workspace (65 files in the release check); nothing is compiled.

## Tests

On the machine above, 2026-11-18 (UTC). The attached `RELEASE_CHECK_rtx40-win.log` has the output (the Windows user
name and local work paths replaced by placeholders), and `fi_rel_check.py` is the check itself (the rtx50-win check
without its XQA parts).

- **Release checks (`fi_rel_check.py`), passed:** a new venv with the torch pre-release, the three wheels and their
  PyPI dependencies, `apache-tvm-ffi==0.1.16`, no MSVC or CUDA toolkit on `PATH`, `CUDA_PATH` unset,
  `FLASHINFER_DISABLE_JIT=1`, an empty workspace, run from an empty directory. The checks:
  - the three versions, torch and apache-tvm-ffi;
  - FA2 paged decode at head dims 64/128/256 in bf16 and fp16 against a float32 reference: max 3.4e-03 (bf16) and
    4.8e-04 (fp16);
  - causal paged prefill, 40 tokens: 7.9e-03 (bf16) and 9.9e-04 (fp16);
  - top-k/top-p sampling returns only top-k ids;
  - nothing was compiled into the workspace;
  - 122 of the 123 DLLs carry this release's build-info stamp.
  The tolerance is 0.02 for every comparison.
- **Not run:** vLLM (see Limitations), upstream's pytest suite, the `tuned/` gates and test suite (bash, which does not
  run natively on Windows), and any multi-GPU test.
