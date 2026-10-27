#!/bin/bash
# tuned/build.sh <variant> — build the flashinfer-python wheel and, for variants that ship one, the
# flashinfer-jit-cache wheel, for one GPU variant (gb10/rtx40/rtx50) and one CUDA toolkit. It creates the
# toolkit's build venv (tuned/build/<cuda-tag>/venv), installs the build and runtime requirements into it, then
# makes both unstamped wheels with tuned/wheel.sh into tuned/build/<cuda-tag>/wheel/. Nothing is installed into the user site or
# ~/.local: the venv is this toolkit's own (the old build installed flashinfer editable into ~/.local, which
# re-pointed whatever flashinfer that held and broke its jit-cache pairing).
#
#   CUDA_TAG=cu134 bash tuned/build.sh gb10
#
# Stage order (docs/ARCHITECTURE.md B3): build.sh -> full_test.sh -> package.sh -> smoke_test.sh ->
# (review and commit the release notes) -> release.sh -> publish.sh.
#
# See tuned/wheel.sh for the label and the two wheels.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/build.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

command -v python3 &>/dev/null || { echo "ERROR: python3 not found on PATH." >&2; exit 1; }
[[ -n "${GPU_TUNED_TORCH_WHEEL_URL}" ]] || { echo "ERROR: tuned/repo.env pins no torch wheel for ${GPU_TUNED_VARIANT}; there is nothing to build against." >&2; exit 1; }

VENV_DIR="${GPU_TUNED_VENV_DIR}"
# Create or reuse (verified), activate, upgrade pip, current setuptools: one shared setup (C9, tuned-common).
gpu_tuned_setup_build_venv "${VENV_DIR}" "${REPO_ROOT}" || exit 1

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
# setuptools comes from gpu_tuned_setup_build_venv above (flashinfer's pyproject needs >=77 for the SPDX license
# field; the common minimum is higher). The old "<82" was torch's ceiling for the stock wheel.
pip install build wheel "packaging>=24.2"
echo "Installing the pinned torch wheel"
pip install "${GPU_TUNED_TORCH_WHEEL_URL}"
echo "Installing flashinfer's requirements (torch stays the pinned wheel)"
grep -v '^torch$' requirements.txt | pip install -r /dev/stdin
# The current setuptools (at least GPU_TUNED_SETUPTOOLS_MIN), installed last so nothing above can hold it back
# (C8, tuned-common). The pinned tuned torch needs only setuptools>=77.0.3.
gpu_tuned_install_setuptools || exit 1

echo "Auditing already-installed tuned packages"
# Non-fatal: surfaces a stock torch that replaced the tuned one, or a stray flashinfer-cubin, before the build relies on it.
gpu_tuned_audit_pinned pip "torch=${GPU_TUNED_VARIANT}"
gpu_tuned_audit_stray pip flashinfer-cubin
echo "Validating the installed torch build-info stamp carries provenance for this variant"
TORCH_CUDA_SO="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda.so"))')"
gpu_tuned_verify_build_info "${TORCH_CUDA_SO}" torch "${GPU_TUNED_VARIANT}" pytorch_build_info

echo "Initializing submodules"
git submodule update --init --recursive

echo ""
bash "${REPO_ROOT}/tuned/wheel.sh" "${GPU_TUNED_ARG_VARIANT}"

echo ""
echo "=========================================="
echo "Build complete (${GPU_TUNED_HW_LABEL})."
echo "=========================================="
echo "Label:        $(cat "${GPU_TUNED_BUILD_ROOT}/release-label")"
echo "Build commit: $(cat "${GPU_TUNED_BUILD_ROOT}/build-commit")"
echo "Next: CUDA_TAG=${CUDA_TAG} bash tuned/full_test.sh ${GPU_TUNED_VARIANT}"
