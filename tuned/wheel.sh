#!/bin/bash
# tuned/wheel.sh <variant> — make the unstamped flashinfer-python wheel and, for variants that ship one, the
# flashinfer-jit-cache wheel, for one GPU variant and one CUDA toolkit, into tuned/build/<cuda-tag>/wheel/, and record
# the release label and the build commit. This is the wheel step of tuned/build.sh, which installs the toolkit's build
# venv first; it refuses to run without that venv. It never publishes, tests or packages (package.sh stamps a copy, H3).
#
#   CUDA_TAG=cu134 bash tuned/wheel.sh gb10     (normally run by tuned/build.sh)
#
# All wheels carry the same local version, +<variant>.cu<cuda>.<YYYYMMDD>[.<seq>] (E1): flashinfer's own
# _check_jit_cache_version refuses a set whose versions differ. The label is allocated once, here (E3/E4): HEAD's UTC
# commit date and the lowest unused sequence among origin's tags. release-label and build-commit are written last, so
# their presence means the build finished.
#
# Since the merge of upstream v0.7.2.post1 the jit-cache is two wheels (upstream's provider-wheel layout): the provider
# flashinfer-jit-cache-<arch> holds the AOT-compiled .so files, scoped to this variant's target models by the
# GPU_TUNED_AOT_* lists in tuned/repo.env (exported here as FLASHINFER_AOT_*; the fork patch in flashinfer-jit-cache-provider/
# build_backend.py maps them onto flashinfer.aot's config), and the flashinfer-jit-cache shim depends on it. With both
# installed and FLASHINFER_DISABLE_JIT=1, a consumer gets MissingJITCacheError for any op outside the set instead of a
# silent compile.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/wheel.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

[[ -d "${GPU_TUNED_VENV_DIR}" ]] || {
    echo "ERROR: ${GPU_TUNED_VENV_DIR} not found. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} (it installs the build venv, then calls this script)." >&2
    exit 1
}
gpu_tuned_verify_venv "${GPU_TUNED_VENV_DIR}" "${REPO_ROOT}"
# shellcheck source=/dev/null
source "${GPU_TUNED_VENV_DIR}/bin/activate"

LOCAL_VERSION="$(gpu_tuned_build_local_version "${GPU_TUNED_VARIANT}" "${REPO_ROOT}")" || exit 1
[[ "${LOCAL_VERSION}" == *".cu${CUDA_VERSION_COMPACT}."* ]] || {
    echo "ERROR: label ${LOCAL_VERSION} disagrees with ${CUDA_TAG} (from ${CUDA_HOME}); nvcc and the toolkit definition must report the same release." >&2
    exit 1
}
LABEL="${LOCAL_VERSION#*.cu${CUDA_VERSION_COMPACT}.}"
echo "Release label: ${LABEL} (local version ${LOCAL_VERSION})"

# Untracked files in the packaged source trees go into the wheel (flashinfer/data/csrc links to csrc/) although the
# build commit does not name them, and the dirty check below sees tracked files only. A 0.7.4 build here shipped 4,727
# stale kernel sources that way. Ignored paths (csrc/generated/, build output) are not checked.
UNTRACKED="$(git status --porcelain --untracked-files=all -- csrc include flashinfer | grep '^??' || true)"
[[ -z "${UNTRACKED}" ]] || {
    echo "ERROR: untracked files in csrc/, include/ or flashinfer/ would be packaged into the wheel ($(printf '%s\n' "${UNTRACKED}" | wc -l) files):" >&2
    printf '%s\n' "${UNTRACKED}" | head -10 >&2
    echo "Remove them (or commit them), then rebuild. List them all with: git status --untracked-files=all -- csrc include flashinfer" >&2
    exit 1
}

# Remove the previous build's markers and wheels first: the markers are rewritten only if this build succeeds.
rm -f "${GPU_TUNED_BUILD_ROOT}/build-commit" "${GPU_TUNED_BUILD_ROOT}/release-label"
mkdir -p "${GPU_TUNED_WHEEL_DIR}"
rm -f "${GPU_TUNED_WHEEL_DIR}"/flashinfer_python-*.whl "${GPU_TUNED_WHEEL_DIR}"/flashinfer_jit_cache-*.whl \
    "${GPU_TUNED_WHEEL_DIR}"/flashinfer_jit_cache_sm*-*.whl
# flashinfer-jit-cache-<arch> is the provider wheel's distribution name, from the dotted arch (12.1a -> sm121a).
PROVIDER_TAG="sm$(printf '%s' "${GPU_TUNED_FLASHINFER_ARCH}" | tr -d '.')"

echo ""
echo "Building the flashinfer-python wheel (+${LOCAL_VERSION})"
# Python source only (py3-none-any, no compiled CUDA binary): it exists so `import flashinfer` works in another venv.
# Kernels JIT-compile on first use unless the flashinfer-jit-cache wheel below is installed beside it.
FLASHINFER_LOCAL_VERSION="${LOCAL_VERSION}" python3 -m build --wheel --no-isolation --outdir "${GPU_TUNED_WHEEL_DIR}" "${REPO_ROOT}"
# The wheel build copies licenses/*.txt to the repo root; clean those up so the working tree stays clean. Upstream
# tracks one of them (LICENSE.kda-for-kda.txt since v0.7.2), so remove only the untracked copies and restore any
# tracked one the copy changed; an unconditional rm made the tree, and so the build commit, dirty.
while IFS= read -r -d '' f; do rm -f -- "${REPO_ROOT}/${f}"; done \
    < <(git -C "${REPO_ROOT}" ls-files --others --exclude-standard -z -- 'LICENSE.*.txt')
while IFS= read -r -d '' f; do git -C "${REPO_ROOT}" checkout -- "${f}"; done \
    < <(git -C "${REPO_ROOT}" ls-files -z -- 'LICENSE.*.txt')

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
        "${REPO_ROOT}"/flashinfer-jit-cache/*.egg-info "${REPO_ROOT}/flashinfer-jit-cache/flashinfer_jit_cache/jit_cache" \
        "${REPO_ROOT}/flashinfer-jit-cache-provider/dist" "${REPO_ROOT}/flashinfer-jit-cache-provider/build" \
        "${REPO_ROOT}"/flashinfer-jit-cache-provider/*.egg-info \
        "${REPO_ROOT}/flashinfer-jit-cache-provider/flashinfer_jit_cache_provider/jit_cache"
    # build/aot-providers/<arch> (ninja's tree) is kept, as build/aot was for the old layout, so a rebuild is incremental.

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
    # The compiled .so files are in the provider wheel (flashinfer-jit-cache-<arch>); flashinfer-jit-cache is the shim that
    # depends on it and discovers it through the flashinfer.jit_cache.providers entry point. Upstream's provider backend
    # compiles for FLASHINFER_JIT_CACHE_PROVIDER_ARCH and, with the fork patch in flashinfer-jit-cache-provider/
    # build_backend.py, honours the FLASHINFER_AOT_* scoping above.
    FLASHINFER_JIT_CACHE_PROVIDER_ARCH="${GPU_TUNED_FLASHINFER_ARCH}" FLASHINFER_LOCAL_VERSION="${LOCAL_VERSION}" \
        python3 -m build --wheel --no-isolation --outdir "${GPU_TUNED_WHEEL_DIR}" "${REPO_ROOT}/flashinfer-jit-cache-provider"
    echo "Building the flashinfer-jit-cache shim wheel (requires flashinfer-jit-cache-${PROVIDER_TAG}==${LOCAL_VERSION})"
    FLASHINFER_JIT_CACHE_WHEEL_KIND=shim FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="${GPU_TUNED_FLASHINFER_ARCH}" \
        FLASHINFER_LOCAL_VERSION="${LOCAL_VERSION}" \
        python3 -m build --wheel --no-isolation --outdir "${GPU_TUNED_WHEEL_DIR}" "${REPO_ROOT}/flashinfer-jit-cache"
fi

MAIN_WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/flashinfer_python-*.whl 2>/dev/null | head -1)"
[[ -n "${MAIN_WHEEL}" ]] || { echo "ERROR: no flashinfer_python wheel in ${GPU_TUNED_WHEEL_DIR}" >&2; exit 1; }
WHEELS=("${MAIN_WHEEL}")
if [[ "${GPU_TUNED_NEEDS_AOT_JIT_CACHE}" == "true" ]]; then
    JIT_WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/flashinfer_jit_cache-*.whl 2>/dev/null | head -1)"
    [[ -n "${JIT_WHEEL}" ]] || { echo "ERROR: no flashinfer_jit_cache wheel in ${GPU_TUNED_WHEEL_DIR}" >&2; exit 1; }
    PROVIDER_WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/flashinfer_jit_cache_${PROVIDER_TAG}-*.whl 2>/dev/null | head -1)"
    [[ -n "${PROVIDER_WHEEL}" ]] || { echo "ERROR: no flashinfer_jit_cache_${PROVIDER_TAG} wheel in ${GPU_TUNED_WHEEL_DIR}" >&2; exit 1; }
    WHEELS+=("${JIT_WHEEL}" "${PROVIDER_WHEEL}")
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
