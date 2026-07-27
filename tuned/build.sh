#!/bin/bash
# tuned/build.sh <variant> — build flashinfer for a single GPU variant
# (gb10/rtx40/rtx50) only, pinning compilation to one CUDA arch instead of
# the full multi-arch matrix used for upstream releases.
#
# Consolidates what used to be three near-identical scripts
# (scripts/build_gb10.sh, scripts/build_rtx40.sh, scripts/build_rtx50.sh)
# into one, parameterized by tuned/devices/<variant>.conf. gb10 remains
# structurally different from rtx40/rtx50 (see the GB10-only block near
# the end) -- it additionally builds a target-model-scoped
# flashinfer-jit-cache wheel for a genuine zero-runtime-JIT deployment;
# rtx40/rtx50 only ship the flashinfer-python wheel (kernels JIT-compile
# lazily on first call, same as flashinfer's normal non-tuned behavior --
# an acceptable tradeoff for interactive desktop use, unlike GB10's
# zero-JIT edge-device goal). No RTX 40/50 target-model scoping has been
# decided yet, so there's nothing equivalent to consolidate there.
#
# See tuned/docs/gb10_build.md for the full GB10 guide.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"
cd "${REPO_ROOT}"

echo "=========================================="
echo "Building flashinfer for ${GPU_TUNED_HW_LABEL} only"
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

echo "::group::Install flashinfer-python (editable, ${GPU_TUNED_VARIANT}-only)"
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

echo "::group::Build flashinfer-python wheel (+${GPU_TUNED_VARIANT} local version) for external consumers, e.g. vllm"
# Python-source-only wheel (py3-none-any tag, no compiled CUDA binary) --
# needed so `import flashinfer` works in another venv. Kernels JIT-compile
# and cache under ~/.cache/flashinfer/ on first use in that venv's process
# (except gb10, which additionally ships a flashinfer-jit-cache wheel below
# for a genuine zero-JIT deployment).
python3 -m pip install --user --upgrade build
rm -rf "${REPO_ROOT}/dist" "${REPO_ROOT}/build" "${REPO_ROOT}"/*.egg-info
FLASHINFER_LOCAL_VERSION="${GPU_TUNED_VARIANT}" python3 -m build --wheel --no-isolation "${REPO_ROOT}"
echo "Built wheel(s):"
ls -lh "${REPO_ROOT}"/dist/*.whl
# the wheel build copies licenses/*.txt to the repo root (to avoid a nested
# path in the wheel); clean those up so the working tree stays clean
rm -f "${REPO_ROOT}"/LICENSE.*.txt
echo "::endgroup::"

if [[ "${GPU_TUNED_VARIANT}" == "gb10" ]]; then
    echo "::group::Build flashinfer-jit-cache wheel (AOT-compiled .so files, GB10-only)"
    # This is the actual binary: flashinfer/jit/core.py's JitSpec.build_and_load()
    # checks FLASHINFER_AOT_DIR (resolved to this package's jit_cache/ dir, see
    # flashinfer/jit/env.py:_get_aot_dir) for a precompiled <name>.so *before*
    # ever invoking JIT/ninja. With this package installed, and
    # FLASHINFER_DISABLE_JIT=1 set in the consumer's environment, vllm gets a
    # hard MissingJITCacheError instead of a silent JIT compile for anything not
    # covered here -- which is what we want, since this build should never JIT.
    # FLASHINFER_CUDA_ARCH_LIST=12.1a (exported by tuned/env.sh) is what keeps
    # this to GB10/SM121a only: flashinfer/aot.py's detect_sm_capabilities()
    # derives which kernel variants to register from the gencode flags actually
    # present for that arch list, so no other-arch entries get compiled in.
    source "${REPO_ROOT}/scripts/jit_cache_build_common.sh"
    compute_jit_cache_parallelism
    echo "MAX_JOBS: ${MAX_JOBS}, FLASHINFER_NVCC_THREADS: ${FLASHINFER_NVCC_THREADS}"
    rm -rf "${REPO_ROOT}/flashinfer-jit-cache/dist" "${REPO_ROOT}/flashinfer-jit-cache/build" \
        "${REPO_ROOT}/flashinfer-jit-cache"/*.egg-info \
        "${REPO_ROOT}/flashinfer-jit-cache/flashinfer_jit_cache/jit_cache"

    # Scoped to our two target models (nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
    # and deepseek-ai/DeepSeek-V4-Flash) instead of flashinfer's full default op
    # matrix -- both are bf16, head_dim=128 dense/GQA attention with fp8/fp4 MoE.
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
    export FLASHINFER_AOT_ADD_MISC="true"    # includes the SSU/Mamba fix, needed for Nemotron-3-Super
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
else
    echo ""
    echo "Build complete."
    echo "To use this build from vllm's venv:"
    echo "  <vllm-venv>/bin/pip install --force-reinstall ${REPO_ROOT}/dist/flashinfer_python-*+${GPU_TUNED_VARIANT}-*.whl"
    echo "  # kernels JIT-compile lazily on first call; no flashinfer-jit-cache wheel"
    echo "  # is built for this variant -- see tuned/build.sh's header comment."
fi
