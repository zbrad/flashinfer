#!/bin/bash
# tuned/env.sh <variant> — device, toolkit and build-environment setup for tuned flashinfer wheels
# (gb10/rtx40/rtx50). Source this file with the variant as $1; do not execute it directly.
#
#   CUDA_TAG=cu134 bash tuned/build.sh gb10     (or CUDA_VER=13.4)
#
# Device facts come from tuned/common-data/devices/<variant>.conf and the CUDA toolkit from
# tuned/common-data/toolkits/<id>.env (both vendored from zbrad/tuned-common by sync-common.sh),
# never from /usr/local/cuda or from what happens to be installed. There is no default: CUDA_TAG or
# CUDA_VER must name one. Repo-specific settings live in tuned/repo.env, loaded after the shared
# definitions; it may not change any shared key, and it may restrict the toolkits this repo builds
# against (gb10: cu134 only).
#
# Exported: GPU_TUNED_VARIANT/PLATFORM/CUDA_ARCH/HW_LABEL (shared), GPU_TUNED_FLASHINFER_ARCH, GPU_TUNED_TORCH_WHEEL_URL,
# GPU_TUNED_NEEDS_AOT_JIT_CACHE and the GPU_TUNED_AOT_* lists (repo.env), CUDA_HOME/CUDA_TAG/CUDA_VER (the toolkit),
# GPU_TUNED_BUILD_ROOT (tuned/build/<cuda-tag>, the one tree for this toolkit: venv/ wheel/ dist/ releases/),
# GPU_TUNED_VENV_DIR, GPU_TUNED_WHEEL_DIR, GPU_TUNED_DIST_DIR, GPU_TUNED_RELEASES_DIR, FLASHINFER_CUDA_ARCH_LIST and
# FLASHINFER_WORKSPACE_BASE (the tree's own JIT workspace, so a build never reads or writes ~/.cache/flashinfer).
#
# Nothing is installed outside the toolkit's venv: the user site and ~/.local are never touched.

GPU_TUNED_ARG_VARIANT="${1:-}"
if [[ -z "${GPU_TUNED_ARG_VARIANT}" ]]; then
    echo "ERROR: env.sh requires a variant argument (gb10/rtx40/rtx50)" >&2
    return 1 2>/dev/null || exit 1
fi

GPU_TUNED_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_TUNED_DATA_DIR="${GPU_TUNED_SELF_DIR}/common-data"
if [[ ! -f "${GPU_TUNED_DATA_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" ]]; then
    echo "ERROR: no device '${GPU_TUNED_ARG_VARIANT}' in ${GPU_TUNED_DATA_DIR}/devices (have: $(cd "${GPU_TUNED_DATA_DIR}/devices" && ls | sed 's/\.conf$//' | tr '\n' ' '))" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck source=common-data/devices/gb10.conf
source "${GPU_TUNED_DATA_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" || return 1 2>/dev/null || exit 1
export GPU_TUNED_VARIANT GPU_TUNED_PLATFORM GPU_TUNED_CUDA_ARCH GPU_TUNED_HW_LABEL

# shellcheck source=common.sh
# Vendored from https://github.com/zbrad/tuned-common (see sync-common.sh).
source "${GPU_TUNED_SELF_DIR}/common.sh" || return 1 2>/dev/null || exit 1

# Fail loudly if this script runs on the wrong host, rather than producing a wrong-architecture wheel.
gpu_tuned_assert_platform "${GPU_TUNED_PLATFORM}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1

# --- CUDA toolkit: CUDA_TAG (cu134) or CUDA_VER (13.4) name one from common-data/toolkits. ---
_fi_tk="${CUDA_TAG:-}"
_fi_req_ver="${CUDA_VER:-}"
if [ -z "${_fi_tk}" ] && [ -n "${_fi_req_ver}" ]; then
    _fi_tk="cu${_fi_req_ver//./}"                           # 13.4 -> cu134
fi
gpu_tuned_load_toolkit "${GPU_TUNED_DATA_DIR}" "${GPU_TUNED_VARIANT}" "${_fi_tk}" || return 1 2>/dev/null || exit 1
if [ -n "${_fi_req_ver}" ] && [ "${_fi_req_ver}" != "${TOOLKIT_CUDA_VER}" ]; then
    echo "[tuned/env] ERROR: CUDA_VER=${_fi_req_ver} conflicts with toolkit ${TOOLKIT_ID} (CUDA ${TOOLKIT_CUDA_VER})." >&2
    return 1 2>/dev/null || exit 1
fi
unset _fi_tk _fi_req_ver

# --- Repo-specific settings (loaded after the shared device and toolkit values; may not change them). ---
gpu_tuned_load_repo_env "${GPU_TUNED_SELF_DIR}/repo.env" || return 1 2>/dev/null || exit 1
# A sourced file's own `return 1` does not stop the loader, so check that repo.env really set its keys.
if [[ -z "${GPU_TUNED_FLASHINFER_ARCH:-}" || -z "${GPU_TUNED_NEEDS_AOT_JIT_CACHE+x}" ]]; then
    echo "ERROR: tuned/repo.env did not set its settings for ${GPU_TUNED_VARIANT}." >&2
    return 1 2>/dev/null || exit 1
fi
export GPU_TUNED_FLASHINFER_ARCH GPU_TUNED_TORCH_WHEEL_URL GPU_TUNED_NEEDS_AOT_JIT_CACHE
export GPU_TUNED_REPO_TOOLKITS GPU_TUNED_REPO_TOOLKIT_REASON

# Which toolkits this variant may be built against for flashinfer (repo.env). Checked before the toolkit is
# verified, so a refused toolkit is reported as the restriction, not as a missing install.
if [[ -n "${GPU_TUNED_REPO_TOOLKITS:-}" && " ${GPU_TUNED_REPO_TOOLKITS} " != *" ${CUDA_TAG} "* ]]; then
    echo "[tuned/env] ERROR: ${GPU_TUNED_VARIANT} flashinfer builds support ${GPU_TUNED_REPO_TOOLKITS} only; ${CUDA_TAG} was asked for." >&2
    echo "  ${GPU_TUNED_REPO_TOOLKIT_REASON}" >&2
    return 1 2>/dev/null || exit 1
fi
gpu_tuned_verify_toolkit || return 1 2>/dev/null || exit 1

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${GPU_TUNED_TOOLKIT_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_PATH="${CUDA_HOME}" CUDACXX="${CUDA_HOME}/bin/nvcc"
# CUDA_VERSION_COMPACT (e.g. "134" for CUDA 13.4) is the toolkit id's digits.
CUDA_VERSION_COMPACT="${CUDA_TAG#cu}"
export CUDA_VERSION_COMPACT

# --- The one build tree for this toolkit (docs/ARCHITECTURE.md C1/C2): tuned/build/<cuda-tag>/. ---
GPU_TUNED_REPO_ROOT="$(cd "${GPU_TUNED_SELF_DIR}/.." && pwd)"
# A tree last used by another variant is refused, never mixed (gpu_tuned_claim_toolkit_tree).
gpu_tuned_claim_toolkit_tree "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1
GPU_TUNED_BUILD_ROOT="${GPU_TUNED_REPO_ROOT}/tuned/build/${CUDA_TAG}"
GPU_TUNED_VENV_DIR="${GPU_TUNED_BUILD_ROOT}/venv"
GPU_TUNED_WHEEL_DIR="${GPU_TUNED_BUILD_ROOT}/wheel"
GPU_TUNED_DIST_DIR="$(gpu_tuned_out_dir dist "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}")"
GPU_TUNED_RELEASES_DIR="$(gpu_tuned_out_dir releases "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}")"
export GPU_TUNED_BUILD_ROOT GPU_TUNED_VENV_DIR GPU_TUNED_WHEEL_DIR GPU_TUNED_DIST_DIR GPU_TUNED_RELEASES_DIR

# One compile arch, from the device file (not the full multi-arch release matrix upstream builds).
export FLASHINFER_CUDA_ARCH_LIST="${GPU_TUNED_FLASHINFER_ARCH}"
# flashinfer's JIT cache lives under the tree, not ~/.cache/flashinfer, so a test run never reuses an op compiled earlier.
export FLASHINFER_WORKSPACE_BASE="${GPU_TUNED_BUILD_ROOT}/fi-workspace"

echo "[tuned/env] GPU_TUNED_VARIANT=${GPU_TUNED_VARIANT} FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST} CUDA_HOME=${CUDA_HOME} tree=${GPU_TUNED_BUILD_ROOT}"
