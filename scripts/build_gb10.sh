#!/bin/bash
# Build flashinfer for GB10 (DGX Spark, SM121a) only.
# This is the gb10-only branch: it pins compilation to a single
# architecture instead of the full multi-arch matrix used for releases.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' \r')
if [ "${COMPUTE_CAP}" != "12.1" ]; then
  echo "ERROR: detected GPU compute capability '${COMPUTE_CAP}', expected '12.1' (GB10/SM121)." >&2
  echo "This script targets the gb10-only branch; use main for other GPUs." >&2
  exit 1
fi

export FLASHINFER_CUDA_ARCH_LIST="12.1a"

echo "=========================================="
echo "Building flashinfer for GB10 (SM121a) only"
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

echo "::group::Install flashinfer-python (editable, GB10-only)"
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

echo ""
echo "Build complete."
