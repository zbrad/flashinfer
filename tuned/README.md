# tuned/ — single-arch flashinfer wheels

Builds the flashinfer-python wheel and, for variants that ship one, the flashinfer-jit-cache wheel (AOT-compiled kernels)
for one GPU variant and one CUDA toolkit. The layout and rules are docs/ARCHITECTURE.md in
[zbrad/tuned-common](https://github.com/zbrad/tuned-common). Only gb10 is built from this repo today (cu134 only);
rtx40 and rtx50 have settings in `repo.env` but are untested.

```
CUDA_TAG=cu134 bash tuned/build.sh gb10        # venv + both unstamped wheels   -> tuned/build/cu134/wheel/
CUDA_TAG=cu134 bash tuned/full_test.sh gb10    # gates + JIT-disabled coverage  -> .../releases/TEST_RESULTS_gb10.log
CUDA_TAG=cu134 bash tuned/package.sh gb10      # stamp, sidecars, DRAFT notes   -> .../dist/
CUDA_TAG=cu134 bash tuned/smoke_test.sh gb10   # the packaged bits run          -> .../releases/SMOKE_RESULTS_gb10.log
# review and commit tuned/releases/RELEASE_NOTES_<version>.md
CUDA_TAG=cu134 bash tuned/release.sh gb10      # gates only; writes the ready-marker, never publishes
CUDA_TAG=cu134 bash tuned/publish.sh gb10      # publishes exactly what release.sh saw
```

| File | Purpose |
|---|---|
| `env.sh` | device, toolkit and tree setup; sourced by every script |
| `repo.env` | flashinfer-only settings: pinned torch wheel, toolkit restriction, the `GPU_TUNED_AOT_*` kernel lists |
| `common.sh`, `sync-common.sh`, `common-data/` | vendored from tuned-common; refresh with `bash tuned/sync-common.sh sync` |
| `build.sh`, `wheel.sh`, `full_test.sh`, `package.sh`, `smoke_test.sh`, `release.sh`, `publish.sh` | the stages above (`build.sh` makes the venv, then `wheel.sh` makes the wheels and the label) |
| `tests/extra_tests.sh` | checks run before a release, outside the gates: vLLM's acceptance test against these wheels with JIT disabled, and a slice of flashinfer's own `tests/` (XQA batch decode for the shipped shapes, Mamba selective state update) |
| `tests/jit_cache_coverage.py` | with `FLASHINFER_DISABLE_JIT=1` and an empty workspace: `trtllm_utils`, every configured XQA shape and a decode kernel must load from the cache, and nothing may compile |
| `docs/gb10_build.md` | why the jit-cache exists, what it covers, how vLLM uses it |
| `docs/shape_survey.md` | which model shapes the jit-cache does and does not cover: a Hugging Face config survey plus small-model runs, with each claim labeled config, measured or inferred |
| `survey/` | the scripts behind that survey (`hf_shape_survey.py`, `run_small_models.py`) and the 2026-10-19 and 2026-10-20 data snapshots in `survey/data/` |
| `compliance.conf` | attestations for the checks `tools/tuned_compliance.sh` cannot make mechanically |

The release label is the UTC commit date of HEAD plus a sequence (`+gb10.cu134.<YYYYMMDD>[.<seq>]`), allocated once by
`build.sh`. Nothing is installed into `~/.local` or the user site.

## Build venv and gates

- The venv is `tuned/build/<cuda-tag>/venv`, made by `gpu_tuned_setup_build_venv` in `build.sh` (`GPU_TUNED_CLEAN_VENV=1`
  recreates it) with the pinned torch wheel from `repo.env`; the current setuptools (at least 84.2) is installed last (C8).
- `package.sh` fails if the venv's setuptools is below the minimum (H10) and audits each wheel's runtime dependencies
  (H9), logged to `tuned/build/<cuda-tag>/dependency-audit-<package>.log`.
