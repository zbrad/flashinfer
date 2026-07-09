#!/bin/bash
# Build flashinfer for RTX 5080/5090 (Blackwell consumer, SM120a) only.
# This is the native-builds branch: it pins compilation to a single
# architecture instead of the full multi-arch matrix used for releases.
# See docs/gb10_build.md for the GB10 (DGX Spark) equivalent of this script.
#
# Unlike build_gb10.sh, this does NOT build a target-model-scoped
# flashinfer-jit-cache wheel: that scoping (FLASHINFER_AOT_* env vars in
# build_gb10.sh) is tied to two specific GB10 target models (Nemotron-3-Super,
# DeepSeek-V4-Flash) with no RTX 40/50 equivalent decided yet. This script
# only builds the flashinfer-python wheel (Python source only, no compiled
# binary -- see docs) with FLASHINFER_CUDA_ARCH_LIST pinned to sm_120a, so
# kernels JIT-compile lazily on first call and cache under
# ~/.cache/flashinfer/, same as flashinfer's normal (non-GB10) behavior. That
# one-time JIT cost at first launch is an acceptable tradeoff for interactive
# desktop use, unlike GB10's zero-JIT edge-device goal.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' \r')
if [ "${COMPUTE_CAP}" != "12.0" ]; then
  echo "ERROR: detected GPU compute capability '${COMPUTE_CAP}', expected '12.0' (RTX 50 / Blackwell consumer)." >&2
  echo "This script targets the native-builds branch; use main for other GPUs." >&2
  exit 1
fi

# "a" suffix targets the Blackwell family-specific SASS variant -- same
# rationale as GB10's 12.1a above -- required for RTX 50-specific tensor
# core instructions not present in the compatible/generic sm_120 target.
export FLASHINFER_CUDA_ARCH_LIST="12.0a"

echo "=========================================="
echo "Building flashinfer for RTX 50 (SM120a) only"
echo "=========================================="
echo "FLASHINFER_CUDA_ARCH_LIST: ${FLASHINFER_CUDA_ARCH_LIST}"
echo "Python: $(python3 --version)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo ""

echo "::group::Initialize submodules"
git submodule update --init --recursive
echo "::endgroup::"

echo "::group::Ensure setuptools/packaging support PEP 639 license metadata"
# Debian/Ubuntu ship setuptools<77 in dist-packages, which can't parse this
# project's SPDX `license = "Apache-2.0"` field in pyproject.toml. Pin to a
# version new enough to parse it but below torch's `setuptools<82` ceiling.
python3 -m pip install --user "setuptools>=77,<82" "packaging>=24.2"
echo "::endgroup::"

echo "::group::Install flashinfer-python (editable, RTX 50-only)"
python3 -m pip install --no-build-isolation -e "${REPO_ROOT}" -v
echo "::endgroup::"

echo "::group::Smoke test: JIT-compile and run a decode kernel"
python3 - <<'PYEOF'
import torch

import flashinfer

assert torch.cuda.is_available(), "CUDA device not available"
major, minor = torch.cuda.get_device_capability(0)
print(f"Device: {torch.cuda.get_device_name(0)} (SM{major}{minor})")

num_qo_heads, num_kv_heads, head_dim, kv_len = 8, 8, 128, 16
q = torch.randn(num_qo_heads, head_dim, dtype=torch.float16, device="cuda")
k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
o = flashinfer.single_decode_with_kv_cache(q, k, v)
assert o.shape == (num_qo_heads, head_dim)
print(f"flashinfer {flashinfer.__version__} smoke test OK: output {o.shape} {o.dtype}")
PYEOF
echo "::endgroup::"

echo "::group::Build flashinfer-python wheel (+rtx50 local version) for external consumers, e.g. vllm"
# Python-source-only wheel (py3-none-any tag, no compiled CUDA binary) --
# needed so `import flashinfer` works in another venv. Kernels JIT-compile
# and cache under ~/.cache/flashinfer/ on first use in that venv's process.
python3 -m pip install --user --upgrade build
rm -rf "${REPO_ROOT}/dist" "${REPO_ROOT}/build" "${REPO_ROOT}"/*.egg-info
FLASHINFER_LOCAL_VERSION=rtx50 python3 -m build --wheel --no-isolation "${REPO_ROOT}"
echo "Built wheel(s):"
ls -lh "${REPO_ROOT}"/dist/*.whl
# the wheel build copies licenses/*.txt to the repo root (to avoid a nested
# path in the wheel); clean those up so the working tree stays clean
rm -f "${REPO_ROOT}"/LICENSE.*.txt
echo "::endgroup::"

echo ""
echo "Build complete."
echo "To use this build from vllm's venv:"
echo "  <vllm-venv>/bin/pip install --force-reinstall ${REPO_ROOT}/dist/flashinfer_python-*+rtx50-*.whl"
echo "  # kernels JIT-compile lazily on first call; no flashinfer-jit-cache wheel"
echo "  # is built by this script -- see the note at the top of this file."
