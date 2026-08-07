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
SHORT_VER="$(echo "${VERSION}" | sed -E 's/^0*([0-9]+)\.0*([0-9]+)\..*/\1.\2/')"

CUDA_VER="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
[[ -z "${CUDA_VER}" ]] && { echo "ERROR: could not determine CUDA version from nvcc." >&2; exit 1; }
CUDA_TAG="cu${CUDA_VER//./}"

MAIN_WHEEL="$(find "${REPO_ROOT}/dist" -maxdepth 1 -name "flashinfer_python-${VERSION}+${GPU_TUNED_VARIANT}-*.whl" | head -1)"
if [[ -z "${MAIN_WHEEL}" ]]; then
    echo "ERROR: no flashinfer_python-${VERSION}+${GPU_TUNED_VARIANT}-*.whl found in dist/." >&2
    echo "  Run 'bash tuned/build.sh ${GPU_TUNED_VARIANT}' first." >&2
    exit 1
fi

ASSETS=("${MAIN_WHEEL}#$(basename "${MAIN_WHEEL}")")
NOTES="flashinfer-python ${VERSION}+${GPU_TUNED_VARIANT} wheel for ${GPU_TUNED_HW_LABEL}, single-arch (sm_${GPU_TUNED_CUDA_ARCH})."

if [[ "${GPU_TUNED_NEEDS_AOT_JIT_CACHE:-false}" == "true" ]]; then
    JIT_WHEEL="$(find "${REPO_ROOT}/flashinfer-jit-cache/dist" -maxdepth 1 -name "flashinfer_jit_cache-${VERSION}+${GPU_TUNED_VARIANT}-*.whl" | head -1)"
    if [[ -z "${JIT_WHEEL}" ]]; then
        echo "ERROR: GPU_TUNED_NEEDS_AOT_JIT_CACHE=true but no flashinfer_jit_cache-${VERSION}+${GPU_TUNED_VARIANT}-*.whl found." >&2
        echo "  Run 'bash tuned/build.sh ${GPU_TUNED_VARIANT}' first." >&2
        exit 1
    fi
    ASSETS+=("${JIT_WHEEL}#$(basename "${JIT_WHEEL}")")
    NOTES="${NOTES} Includes flashinfer-jit-cache (AOT-compiled kernels, no runtime JIT for its covered ops -- see tuned/devices/${GPU_TUNED_VARIANT}.conf for exact model/op coverage)."
else
    NOTES="${NOTES} Kernels JIT-compile lazily on first use in the consuming environment; no AOT jit-cache wheel for this variant."
fi

RELEASE_TAG="v${SHORT_VER}-${GPU_TUNED_VARIANT}-${CUDA_TAG}"
RELEASE_TITLE="flashinfer ${SHORT_VER} — ${GPU_TUNED_HW_LABEL} (${CUDA_TAG})"

echo "===================================================="
echo "flashinfer ${GPU_TUNED_HW_LABEL} Release"
echo "===================================================="
echo "  Version : ${VERSION} (short: ${SHORT_VER})"
echo "  CUDA    : ${CUDA_TAG}"
echo "  Assets  :"
for a in "${ASSETS[@]}"; do echo "    - ${a%%#*}"; done
echo ""
echo "Publishing to GitHub release ${RELEASE_TAG}..."
gh release create "${RELEASE_TAG}" \
    --repo zbrad/flashinfer \
    --title "${RELEASE_TITLE}" \
    --target "tuned-builds" \
    --notes "${NOTES}" \
    "${ASSETS[@]}"

echo ""
echo "Release: https://github.com/zbrad/flashinfer/releases/tag/${RELEASE_TAG}"
echo "Done."
