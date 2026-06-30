#!/bin/bash
# Build flashinfer for GB10 (DGX Spark, SM121a) only.
# This is the gb10-only branch: it pins compilation to a single
# architecture instead of the full multi-arch matrix used for releases.
# See docs/gb10_build.md for the full guide, including the vLLM-side steps
# needed to actually route inference through these AOT-cached kernels.
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

echo "::group::Build flashinfer-python wheel (+gb10 local version) for external consumers, e.g. vllm"
# This wheel only carries Python source + csrc/include — no compiled CUDA
# binary (confirmed by its py3-none-any tag). It's needed so `import
# flashinfer` works in another venv; the actual GB10 binaries come from the
# flashinfer-jit-cache wheel built below. FLASHINFER_LOCAL_VERSION=gb10 marks
# it as the GB10-only build, following the convention release.yml uses for
# CUDA version suffixes (e.g. "+cu128").
python3 -m pip install --user --upgrade build
rm -rf "${REPO_ROOT}/dist" "${REPO_ROOT}/build" "${REPO_ROOT}"/*.egg-info
FLASHINFER_LOCAL_VERSION=gb10 python3 -m build --wheel --no-isolation "${REPO_ROOT}"
echo "Built wheel(s):"
ls -lh "${REPO_ROOT}"/dist/*.whl
# the wheel build copies licenses/*.txt to the repo root (to avoid a nested
# path in the wheel); clean those up so the working tree stays clean
rm -f "${REPO_ROOT}"/LICENSE.*.txt
echo "::endgroup::"

echo "::group::Build flashinfer-jit-cache wheel (AOT-compiled .so files, GB10-only)"
# This is the actual binary: flashinfer/jit/core.py's JitSpec.build_and_load()
# checks FLASHINFER_AOT_DIR (resolved to this package's jit_cache/ dir, see
# flashinfer/jit/env.py:_get_aot_dir) for a precompiled <name>.so *before*
# ever invoking JIT/ninja. With this package installed, and
# FLASHINFER_DISABLE_JIT=1 set in the consumer's environment, vllm gets a
# hard MissingJITCacheError instead of a silent JIT compile for anything not
# covered here — which is what we want, since this build should never JIT.
# FLASHINFER_CUDA_ARCH_LIST=12.1a (exported above) is what keeps this to
# GB10/SM121a only: flashinfer/aot.py's detect_sm_capabilities() derives
# which kernel variants to register from the gencode flags actually present
# for that arch list, so no other-arch entries get compiled in.
source "${SCRIPT_DIR}/jit_cache_build_common.sh"
compute_jit_cache_parallelism
echo "MAX_JOBS: ${MAX_JOBS}, FLASHINFER_NVCC_THREADS: ${FLASHINFER_NVCC_THREADS}"
rm -rf "${REPO_ROOT}/flashinfer-jit-cache/dist" "${REPO_ROOT}/flashinfer-jit-cache/build" \
  "${REPO_ROOT}/flashinfer-jit-cache"/*.egg-info \
  "${REPO_ROOT}/flashinfer-jit-cache/flashinfer_jit_cache/jit_cache"

# Scoped to our two target models (nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
# and deepseek-ai/DeepSeek-V4-Flash) instead of flashinfer's full default op
# matrix — both are bf16, head_dim=128 dense/GQA attention with fp8/fp4 MoE.
# DeepSeek-V4's MLA path goes through gen_sparse_mla_sm120_module(), which is
# unconditional for SM120/121 and unaffected by these flags. FA3 (192,128) is
# kept alongside (128,128) as a safety margin for any legacy-shaped MLA
# fallback. See flashinfer-jit-cache/build_backend.py:_config_overrides_from_env
# for how these map onto flashinfer.aot's config (env var names mirror
# flashinfer/aot.py's CLI flags).
export FLASHINFER_AOT_FA2_HEAD_DIM="128,128"
export FLASHINFER_AOT_FA3_HEAD_DIM="128,128 192,128"
export FLASHINFER_AOT_F16_DTYPE="bfloat16"
export FLASHINFER_AOT_ADD_COMM="false"   # single GPU box, no NCCL-style collectives
export FLASHINFER_AOT_ADD_GEMMA="false"  # neither target model is Gemma
export FLASHINFER_AOT_ADD_OAI_OSS="false" # neither target model is gpt-oss
export FLASHINFER_AOT_ADD_MOE="true"     # both targets are MoE (fp4/fp8)
export FLASHINFER_AOT_ADD_ACT="true"
export FLASHINFER_AOT_ADD_MISC="true"    # includes the SSU/Mamba fix above, needed for Nemotron-3-Super
export FLASHINFER_AOT_ADD_XQA="false"    # not used by either target model on this backend
FLASHINFER_LOCAL_VERSION=gb10 python3 -m build --wheel --no-isolation "${REPO_ROOT}/flashinfer-jit-cache"
echo "Built wheel(s):"
ls -lh "${REPO_ROOT}"/flashinfer-jit-cache/dist/*.whl
echo "::endgroup::"

echo ""
echo "Build complete."
echo "To use this build from vllm's venv (no JIT at runtime):"
echo "  <vllm-venv>/bin/pip install --force-reinstall \\"
echo "    ${REPO_ROOT}/dist/flashinfer_python-*+gb10-*.whl \\"
echo "    ${REPO_ROOT}/flashinfer-jit-cache/dist/flashinfer_jit_cache-*+gb10-*.whl"
echo "  # then, in vllm's environment, set FLASHINFER_DISABLE_JIT=1 to turn any"
echo "  # cache-miss op into a hard error instead of a silent JIT compile."
