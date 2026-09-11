#!/bin/bash
# tuned/release.sh <variant> — publish an already-built (tuned/build.sh
# <variant>) flashinfer-python wheel (and, for variants with
# GPU_TUNED_NEEDS_AOT_JIT_CACHE=true, the flashinfer-jit-cache wheel) as a
# real GitHub release, matching zbrad/raft's/zbrad/cuvs's/zbrad/faiss's
# tuned/package.sh conventions (v<short_ver>-<variant>-<cuda_tag> tag,
# --target tuned-builds). flashinfer previously built wheels into dist/
# with no publish step at all.
#
# CUDA_TAG here is informational only (flashinfer-python's wheel itself is
# pure-Python, py3-none-any -- it doesn't encode a CUDA version in its own
# filename the way raft/cuvs/faiss .so's do); derived from nvcc so the
# release tag still records which toolkit the AOT jit-cache .so (if any)
# was compiled against.
#
# Usage:
#   bash tuned/build.sh gb10      # first, produces dist/*.whl (+ jit-cache)
#   bash tuned/release.sh gb10    # then, publish both as one release
set -euo pipefail

GPU_TUNED_ARG_VARIANT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"
cd "${REPO_ROOT}"

VERSION="$(tr -d '\r' < "${REPO_ROOT}/version.txt")"

# Wildcard after ${GPU_TUNED_VARIANT} (not a literal "-") -- build.sh's
# FLASHINFER_LOCAL_VERSION now carries variant.cuTAG.tuning-vN, not just
# the bare variant, so the wheel filename has more after it than a single
# "-cp314-..." tag suffix.
MAIN_WHEEL="$(find "${REPO_ROOT}/dist" -maxdepth 1 -name "flashinfer_python-${VERSION}+${GPU_TUNED_VARIANT}*.whl" | head -1)"
if [[ -z "${MAIN_WHEEL}" ]]; then
    echo "ERROR: no flashinfer_python-${VERSION}+${GPU_TUNED_VARIANT}*.whl found in dist/." >&2
    echo "  Run 'bash tuned/build.sh ${GPU_TUNED_VARIANT}' first." >&2
    exit 1
fi

# Read the full local version (variant, cuda tag, tuning-vN) back from the
# wheel filename rather than re-deriving cuda tag independently via nvcc --
# avoids any drift between what build.sh actually baked in and what this
# script assumes it did.
FULL_VERSION="$(gpu_tuned_wheel_version "${MAIN_WHEEL}" flashinfer_python)" || exit 1
GIT_SHA="$(git rev-parse --short HEAD)"
CUDA_TAG="$(echo "${FULL_VERSION}" | grep -oE 'cu[0-9]+' | head -1)"
TUNING_LABEL="$(echo "${FULL_VERSION}" | grep -oE 'tuning-v[0-9]+' | head -1)"
[[ -z "${CUDA_TAG}" ]] && { echo "ERROR: could not determine CUDA tag from wheel version ${FULL_VERSION}." >&2; exit 1; }

ASSETS=("${MAIN_WHEEL}#$(basename "${MAIN_WHEEL}")")
NOTES="flashinfer-python ${FULL_VERSION} wheel for ${GPU_TUNED_HW_LABEL}, single-arch (sm_${GPU_TUNED_CUDA_ARCH})."

if [[ "${GPU_TUNED_NEEDS_AOT_JIT_CACHE:-false}" == "true" ]]; then
    JIT_WHEEL="$(find "${REPO_ROOT}/flashinfer-jit-cache/dist" -maxdepth 1 -name "flashinfer_jit_cache-${VERSION}+${GPU_TUNED_VARIANT}*.whl" | head -1)"
    if [[ -z "${JIT_WHEEL}" ]]; then
        echo "ERROR: GPU_TUNED_NEEDS_AOT_JIT_CACHE=true but no flashinfer_jit_cache-${VERSION}+${GPU_TUNED_VARIANT}*.whl found." >&2
        echo "  Run 'bash tuned/build.sh ${GPU_TUNED_VARIANT}' first." >&2
        exit 1
    fi
    ASSETS+=("${JIT_WHEEL}#$(basename "${JIT_WHEEL}")")
    NOTES="${NOTES} Includes flashinfer-jit-cache (AOT-compiled kernels, no runtime JIT for its covered ops -- see tuned/devices/${GPU_TUNED_VARIANT}.conf for exact model/op coverage)."
else
    NOTES="${NOTES} Kernels JIT-compile lazily on first use in the consuming environment; no AOT jit-cache wheel for this variant."
fi

# RELEASE_TAG carries the full local version (the wheel filename's own
# identity); RELEASE_TITLE is the human-friendly separation of upstream
# version / tuned build identity / traceability, same shape as every
# other tuned-builds repo's release.sh|wheel.sh this session.
RELEASE_TAG="v${FULL_VERSION}"
RELEASE_TITLE="flashinfer ${VERSION} — ${GPU_TUNED_VARIANT} ${TUNING_LABEL:-(pre-tuning-v build)} (${CUDA_TAG}, ${GIT_SHA})"

echo "===================================================="
echo "flashinfer ${GPU_TUNED_HW_LABEL} Release"
echo "===================================================="
echo "  Version : ${FULL_VERSION}"
echo "  CUDA    : ${CUDA_TAG}"
echo "  Assets  :"
for a in "${ASSETS[@]}"; do echo "    - ${a%%#*}"; done
echo ""
echo "Publishing to GitHub release ${RELEASE_TAG}..."
gpu_tuned_publish_release "zbrad/flashinfer" "${RELEASE_TAG}" "${RELEASE_TITLE}" \
    "${NOTES}" "${ASSETS[@]}"

echo ""
echo "Release: https://github.com/zbrad/flashinfer/releases/tag/${RELEASE_TAG}"
echo "Done."
