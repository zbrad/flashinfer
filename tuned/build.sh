#!/bin/bash
# tuned/build.sh <variant> — build the flashinfer-python wheel and, for variants that ship one, the
# flashinfer-jit-cache wheel, for one GPU variant (gb10/rtx40/rtx50) and one CUDA toolkit. It creates the
# toolkit's build venv (tuned/build/<cuda-tag>/venv), installs the build and runtime requirements into it, then
# builds both unstamped wheels into tuned/build/<cuda-tag>/wheel/. Nothing is installed into the user site or
# ~/.local: the venv is this toolkit's own (the old build installed flashinfer editable into ~/.local, which
# re-pointed whatever flashinfer that held and broke its jit-cache pairing).
#
#   CUDA_TAG=cu134 bash tuned/build.sh gb10
#
# Stage order (docs/ARCHITECTURE.md B3): build.sh -> full_test.sh -> package.sh -> smoke_test.sh ->
# (review and commit the release notes) -> release.sh -> publish.sh.
#
# Both wheels carry the same local version, +<variant>.cu<cuda>.<YYYYMMDD>[.<seq>] (E1): flashinfer's own
# _check_jit_cache_version refuses a pair whose versions differ. The label is allocated once, here (E3/E4), and
# written to release-label with the build commit; later stages read it and never recompute it.
#
# flashinfer-jit-cache is the actual binary: AOT-compiled .so files, scoped to this variant's target models by
# the GPU_TUNED_AOT_* lists in tuned/repo.env (flashinfer-jit-cache/build_backend.py reads them as
# FLASHINFER_AOT_* and maps them onto flashinfer.aot's config). With it installed and FLASHINFER_DISABLE_JIT=1, a
# consumer gets MissingJITCacheError for any op outside the set instead of a silent compile.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/build.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

command -v python3 &>/dev/null || { echo "ERROR: python3 not found on PATH." >&2; exit 1; }
[[ -n "${GPU_TUNED_TORCH_WHEEL_URL}" ]] || { echo "ERROR: tuned/repo.env pins no torch wheel for ${GPU_TUNED_VARIANT}; there is nothing to build against." >&2; exit 1; }

VENV_DIR="${GPU_TUNED_VENV_DIR}"
if [[ "${GPU_TUNED_CLEAN_VENV:-}" == "1" && -d "${VENV_DIR}" ]]; then
    echo "GPU_TUNED_CLEAN_VENV=1: removing ${VENV_DIR} for a clean rebuild"
    rm -rf "${VENV_DIR}"
fi
if [[ -d "${VENV_DIR}" ]]; then
    # Reusing the venv is the default: the install is slow. Verify it was not copied from another checkout.
    gpu_tuned_verify_venv "${VENV_DIR}" "${REPO_ROOT}"
else
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "=========================================="
echo "Building flashinfer for ${GPU_TUNED_HW_LABEL}"
echo "=========================================="
echo "Toolkit:                  ${CUDA_TAG} (${CUDA_HOME})"
echo "Venv:                     ${VENV_DIR}"
echo "FLASHINFER_CUDA_ARCH_LIST: ${FLASHINFER_CUDA_ARCH_LIST}"
echo "Python:                   $(python3 --version)"
echo "Git commit:               $(git rev-parse HEAD)"
echo ""

echo "Installing build-time requirements..."
pip install --upgrade pip
# setuptools>=77 parses the SPDX license field in pyproject.toml; torch's ceiling is <82.
pip install build wheel "setuptools>=77,<82" "packaging>=24.2"
echo "Installing the pinned torch wheel"
pip install "${GPU_TUNED_TORCH_WHEEL_URL}"
echo "Installing flashinfer's requirements (torch stays the pinned wheel)"
grep -v '^torch$' requirements.txt | pip install -r /dev/stdin

echo "Auditing already-installed tuned packages"
# Non-fatal: surfaces a stock torch that replaced the tuned one, or a stray flashinfer-cubin, before the build relies on it.
gpu_tuned_audit_pinned pip "torch=${GPU_TUNED_VARIANT}"
gpu_tuned_audit_stray pip flashinfer-cubin
echo "Validating the installed torch build-info stamp carries provenance for this variant"
TORCH_CUDA_SO="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda.so"))')"
gpu_tuned_verify_build_info "${TORCH_CUDA_SO}" torch "${GPU_TUNED_VARIANT}" pytorch_build_info

echo "Initializing submodules"
git submodule update --init --recursive

LOCAL_VERSION="$(gpu_tuned_build_local_version "${GPU_TUNED_VARIANT}" "${REPO_ROOT}")" || exit 1
[[ "${LOCAL_VERSION}" == *".cu${CUDA_VERSION_COMPACT}."* ]] || {
    echo "ERROR: label ${LOCAL_VERSION} disagrees with ${CUDA_TAG} (from ${CUDA_HOME}); nvcc and the toolkit definition must report the same release." >&2
    exit 1
}
LABEL="${LOCAL_VERSION#*.cu${CUDA_VERSION_COMPACT}.}"
echo "Release label: ${LABEL} (local version ${LOCAL_VERSION})"

# Remove the previous build's markers and wheels first: the markers are rewritten only if this build succeeds.
rm -f "${GPU_TUNED_BUILD_ROOT}/build-commit" "${GPU_TUNED_BUILD_ROOT}/release-label"
mkdir -p "${GPU_TUNED_WHEEL_DIR}"
rm -f "${GPU_TUNED_WHEEL_DIR}"/flashinfer_python-*.whl "${GPU_TUNED_WHEEL_DIR}"/flashinfer_jit_cache-*.whl

echo ""
echo "Building the flashinfer-python wheel (+${LOCAL_VERSION})"
# Python source only (py3-none-any, no compiled CUDA binary): it exists so `import flashinfer` works in another venv.
# Kernels JIT-compile on first use unless the flashinfer-jit-cache wheel below is installed beside it.
FLASHINFER_LOCAL_VERSION="${LOCAL_VERSION}" python3 -m build --wheel --no-isolation --outdir "${GPU_TUNED_WHEEL_DIR}" "${REPO_ROOT}"
# The wheel build copies licenses/*.txt to the repo root; clean those up so the working tree stays clean.
rm -f "${REPO_ROOT}"/LICENSE.*.txt

if [[ "${GPU_TUNED_NEEDS_AOT_JIT_CACHE}" == "true" ]]; then
    echo ""
    echo "Building the flashinfer-jit-cache wheel (AOT-compiled .so files, ${GPU_TUNED_VARIANT} only)"
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/scripts/jit_cache_build_common.sh"
    # Compile jobs: gpu_tuned_jobs unless AOT_MAX_JOBS_CAP is set (C5); the helper then also budgets memory per job.
    export AOT_MAX_JOBS_CAP="${AOT_MAX_JOBS_CAP:-$(gpu_tuned_jobs)}"
    compute_jit_cache_parallelism
    echo "MAX_JOBS: ${MAX_JOBS}, FLASHINFER_NVCC_THREADS: ${FLASHINFER_NVCC_THREADS}"
    rm -rf "${REPO_ROOT}/flashinfer-jit-cache/dist" "${REPO_ROOT}/flashinfer-jit-cache/build" \
        "${REPO_ROOT}"/flashinfer-jit-cache/*.egg-info "${REPO_ROOT}/flashinfer-jit-cache/flashinfer_jit_cache/jit_cache"

    export FLASHINFER_AOT_FA2_HEAD_DIM="${GPU_TUNED_AOT_FA2_HEAD_DIM}"
    export FLASHINFER_AOT_FA3_HEAD_DIM="${GPU_TUNED_AOT_FA3_HEAD_DIM}"
    export FLASHINFER_AOT_F16_DTYPE="${GPU_TUNED_AOT_F16_DTYPE}"
    export FLASHINFER_AOT_ADD_COMM="${GPU_TUNED_AOT_ADD_COMM}"
    export FLASHINFER_AOT_ADD_GEMMA="${GPU_TUNED_AOT_ADD_GEMMA}"
    export FLASHINFER_AOT_ADD_OAI_OSS="${GPU_TUNED_AOT_ADD_OAI_OSS}"
    export FLASHINFER_AOT_ADD_MOE="${GPU_TUNED_AOT_ADD_MOE}"
    export FLASHINFER_AOT_ADD_ACT="${GPU_TUNED_AOT_ADD_ACT}"
    export FLASHINFER_AOT_ADD_MISC="${GPU_TUNED_AOT_ADD_MISC}"
    export FLASHINFER_AOT_ADD_XQA="${GPU_TUNED_AOT_ADD_XQA}"
    # Optional XQA shape lists (flashinfer/aot.py); unset keeps upstream's defaults.
    [[ -n "${GPU_TUNED_AOT_XQA_INPUT_TYPE:-}" ]] && export FLASHINFER_AOT_XQA_INPUT_TYPE="${GPU_TUNED_AOT_XQA_INPUT_TYPE}"
    [[ -n "${GPU_TUNED_AOT_XQA_FP8_KV_CACHE:-}" ]] && export FLASHINFER_AOT_XQA_FP8_KV_CACHE="${GPU_TUNED_AOT_XQA_FP8_KV_CACHE}"
    [[ -n "${GPU_TUNED_AOT_XQA_TOKEN_PER_PAGE:-}" ]] && export FLASHINFER_AOT_XQA_TOKEN_PER_PAGE="${GPU_TUNED_AOT_XQA_TOKEN_PER_PAGE}"
    [[ -n "${GPU_TUNED_AOT_XQA_HEAD_SIZE:-}" ]] && export FLASHINFER_AOT_XQA_HEAD_SIZE="${GPU_TUNED_AOT_XQA_HEAD_SIZE}"
    [[ -n "${GPU_TUNED_AOT_XQA_HEAD_GRP_SIZE:-}" ]] && export FLASHINFER_AOT_XQA_HEAD_GRP_SIZE="${GPU_TUNED_AOT_XQA_HEAD_GRP_SIZE}"
    FLASHINFER_LOCAL_VERSION="${LOCAL_VERSION}" python3 -m build --wheel --no-isolation --outdir "${GPU_TUNED_WHEEL_DIR}" "${REPO_ROOT}/flashinfer-jit-cache"
fi

MAIN_WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/flashinfer_python-*.whl 2>/dev/null | head -1)"
[[ -n "${MAIN_WHEEL}" ]] || { echo "ERROR: no flashinfer_python wheel in ${GPU_TUNED_WHEEL_DIR}" >&2; exit 1; }
WHEELS=("${MAIN_WHEEL}")
if [[ "${GPU_TUNED_NEEDS_AOT_JIT_CACHE}" == "true" ]]; then
    JIT_WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/flashinfer_jit_cache-*.whl 2>/dev/null | head -1)"
    [[ -n "${JIT_WHEEL}" ]] || { echo "ERROR: no flashinfer_jit_cache wheel in ${GPU_TUNED_WHEEL_DIR}" >&2; exit 1; }
    WHEELS+=("${JIT_WHEEL}")
fi
for w in "${WHEELS[@]}"; do
    pkg="$(basename "${w}")"; pkg="${pkg%%-*}"
    VERSION="$(gpu_tuned_wheel_version "${w}" "${pkg}")" || exit 1
    [[ "${VERSION}" == *"+${LOCAL_VERSION}" ]] || {
        echo "ERROR: $(basename "${w}") is version ${VERSION}, not carrying +${LOCAL_VERSION} from the release label." >&2
        exit 1
    }
    gpu_tuned_check_wheel_label "${w}"
    echo "Built wheel: $(basename "${w}") ($(du -sh "${w}" | awk '{print $1}'))"
done

# build-commit: the source the wheels were made from. A dirty tree is recorded as such and package.sh refuses it.
BUILD_COMMIT="$(git rev-parse HEAD)"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || BUILD_COMMIT="${BUILD_COMMIT}-dirty"
printf '%s\n' "${LABEL}" > "${GPU_TUNED_BUILD_ROOT}/release-label"
printf '%s\n' "${BUILD_COMMIT}" > "${GPU_TUNED_BUILD_ROOT}/build-commit"

echo ""
echo "=========================================="
echo "Build complete (${GPU_TUNED_HW_LABEL})."
echo "=========================================="
echo "Label:        ${LABEL}"
echo "Build commit: ${BUILD_COMMIT}"
echo "Next: CUDA_TAG=${CUDA_TAG} bash tuned/full_test.sh ${GPU_TUNED_VARIANT}"
