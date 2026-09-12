#!/bin/bash
# tuned/build.sh <variant> — build flashinfer for a single GPU variant
# (gb10/rtx40/rtx50) only, pinning compilation to one CUDA arch instead of
# the full multi-arch matrix used for upstream releases.
#
# Consolidates what used to be three near-identical scripts
# (scripts/build_gb10.sh, scripts/build_rtx40.sh, scripts/build_rtx50.sh)
# into one, parameterized by tuned/devices/<variant>.conf. gb10 and rtx50
# both additionally build a target-model-scoped flashinfer-jit-cache wheel
# for a (nominally) zero-runtime-JIT deployment -- see
# GPU_TUNED_NEEDS_AOT_JIT_CACHE and each variant's own
# tuned/devices/<variant>.conf for its specific target model(s) and AOT
# flag values. rtx40 has no target-model scoping decided, so it only ships
# the flashinfer-python wheel (kernels JIT-compile lazily on first call,
# same as flashinfer's normal non-tuned behavior).
#
# See tuned/docs/gb10_build.md for the full GB10 guide.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"
cd "${REPO_ROOT}"

# tuning-vN = commits on tuned-builds since it diverged from main (i.e.
# commits ahead of upstream/flashinfer-ai) -- same convention adopted
# fleet-wide from zbrad/pytorch's tuned/wheel.sh: version.txt's plain
# semver only moves when upstream bumps it, so on its own it can't say
# "how much of our own tuned-builds work landed since an earlier wheel
# was built." flashinfer-python and flashinfer-jit-cache must carry the
# exact same local version (flashinfer's own _check_jit_cache_version
# enforces this at import time), so this is computed once and reused for
# both builds below.
gpu_tuned_resolve_cuda_home
TUNED_COMMIT_COUNT="$(git rev-list --count main..HEAD)"
FLASHINFER_TUNED_LOCAL_VERSION="${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.tuning-v${TUNED_COMMIT_COUNT}"

echo "=========================================="
echo "Building flashinfer for ${GPU_TUNED_HW_LABEL} only"
echo "=========================================="
echo "FLASHINFER_CUDA_ARCH_LIST: ${FLASHINFER_CUDA_ARCH_LIST}"
echo "Python: $(python3 --version)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo ""

# This build installs into the shared ~/.local (--user), not a per-repo
# venv (flashinfer's build backend doesn't work inside one -- see
# tuned/docs/gb10_build.md) -- so unlike the fleet's other tuned/build.sh
# scripts, there's no dedicated venv to verify or recreate. Instead, audit
# what's already there before installing on top of it: flag a torch not
# carrying this fleet's local-version tag (a plain PyPI torch could have
# silently landed via some other tool's install into the same ~/.local),
# and flag known stray packages that break flashinfer's own runtime
# version checks just by being present (see the flashinfer-cubin
# incident -- an unrelated PyPI package whose name collides with part of
# flashinfer's own cubin-loading path).
echo "::group::Audit ~/.local before installing"
gpu_tuned_audit_pinned "python3 -m pip" "torch=${GPU_TUNED_VARIANT}"
gpu_tuned_audit_stray "python3 -m pip" flashinfer-cubin
echo "::endgroup::"

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

echo "::group::Build flashinfer-python wheel (+${FLASHINFER_TUNED_LOCAL_VERSION} local version) for external consumers, e.g. vllm"
# Python-source-only wheel (py3-none-any tag, no compiled CUDA binary) --
# needed so `import flashinfer` works in another venv. Kernels JIT-compile
# and cache under ~/.cache/flashinfer/ on first use in that venv's process
# (except variants with GPU_TUNED_NEEDS_AOT_JIT_CACHE=true, which
# additionally ship a flashinfer-jit-cache wheel below for a nominally
# zero-JIT deployment).
python3 -m pip install --user --upgrade build
rm -rf "${REPO_ROOT}/dist" "${REPO_ROOT}/build" "${REPO_ROOT}"/*.egg-info
FLASHINFER_LOCAL_VERSION="${FLASHINFER_TUNED_LOCAL_VERSION}" python3 -m build --wheel --no-isolation "${REPO_ROOT}"
echo "Built wheel(s):"
ls -lh "${REPO_ROOT}"/dist/*.whl
# the wheel build copies licenses/*.txt to the repo root (to avoid a nested
# path in the wheel); clean those up so the working tree stays clean
rm -f "${REPO_ROOT}"/LICENSE.*.txt
echo "::endgroup::"

if [[ "${GPU_TUNED_NEEDS_AOT_JIT_CACHE}" == "true" ]]; then
    echo "::group::Build flashinfer-jit-cache wheel (AOT-compiled .so files, ${GPU_TUNED_VARIANT}-only)"
    # This is the actual binary: flashinfer/jit/core.py's JitSpec.build_and_load()
    # checks FLASHINFER_AOT_DIR (resolved to this package's jit_cache/ dir, see
    # flashinfer/jit/env.py:_get_aot_dir) for a precompiled <name>.so *before*
    # ever invoking JIT/ninja. With this package installed, and
    # FLASHINFER_DISABLE_JIT=1 set in the consumer's environment, vllm gets a
    # hard MissingJITCacheError instead of a silent JIT compile for anything not
    # covered here -- which is what we want, since this build should never JIT.
    # FLASHINFER_CUDA_ARCH_LIST (exported by tuned/env.sh) is what keeps this
    # to a single arch: flashinfer/aot.py's detect_sm_capabilities() derives
    # which kernel variants to register from the gencode flags actually
    # present for that arch list, so no other-arch entries get compiled in.
    source "${REPO_ROOT}/scripts/jit_cache_build_common.sh"
    compute_jit_cache_parallelism
    echo "MAX_JOBS: ${MAX_JOBS}, FLASHINFER_NVCC_THREADS: ${FLASHINFER_NVCC_THREADS}"
    rm -rf "${REPO_ROOT}/flashinfer-jit-cache/dist" "${REPO_ROOT}/flashinfer-jit-cache/build" \
        "${REPO_ROOT}/flashinfer-jit-cache"/*.egg-info \
        "${REPO_ROOT}/flashinfer-jit-cache/flashinfer_jit_cache/jit_cache"

    # Scoped to this variant's target model(s) instead of flashinfer's full
    # default op matrix -- see tuned/devices/<variant>.conf's
    # GPU_TUNED_AOT_* values and comments for exactly which models and why.
    # See flashinfer-jit-cache/build_backend.py:_config_overrides_from_env
    # for how these map onto flashinfer.aot's config (env var names mirror
    # flashinfer/aot.py's CLI flags).
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
    FLASHINFER_LOCAL_VERSION="${FLASHINFER_TUNED_LOCAL_VERSION}" python3 -m build --wheel --no-isolation "${REPO_ROOT}/flashinfer-jit-cache"
    echo "Built wheel(s):"
    ls -lh "${REPO_ROOT}"/flashinfer-jit-cache/dist/*.whl
    echo "::endgroup::"

    echo "::group::Stamp build-info into every AOT-compiled kernel .so"
    # Same proven-safe pattern as pytorch/flash-attention/flash-attention-vllm's
    # wheel.sh this session: python -m build's own install pass isn't
    # guaranteed to carry forward a stamp added to a pre-build .so, so stamp
    # the wheel's own contents post-build, verify each one survived, then
    # repack (regenerates RECORD correctly, unlike a raw zip edit). This
    # wheel packages many independently-compiled kernel .so files (one per
    # AOT-covered op signature, see GPU_TUNED_AOT_* above) rather than one
    # primary binary -- stamp and verify all of them, not just one, so any
    # individual kernel's provenance is checkable on its own later.
    JIT_WHEEL="$(find "${REPO_ROOT}/flashinfer-jit-cache/dist" -maxdepth 1 -name 'flashinfer_jit_cache-*.whl' | head -1)"
    [[ -z "${JIT_WHEEL}" ]] && { echo "ERROR: no flashinfer-jit-cache wheel found to stamp." >&2; exit 1; }
    JIT_VERSION="$(gpu_tuned_wheel_version "${JIT_WHEEL}" flashinfer_jit_cache)" || exit 1
    python3 -m pip install --user --upgrade wheel >/dev/null
    UNPACK_DIR="$(mktemp -d)"
    python3 -m wheel unpack "${JIT_WHEEL}" --dest "${UNPACK_DIR}"
    mapfile -t JIT_SOS < <(find "${UNPACK_DIR}" -name '*.so')
    [[ ${#JIT_SOS[@]} -eq 0 ]] && { echo "ERROR: no .so files found inside ${JIT_WHEEL}." >&2; exit 1; }
    echo "Stamping ${#JIT_SOS[@]} compiled kernel(s) with build-info"
    for so in "${JIT_SOS[@]}"; do
        embed_build_info "${so}" "${GPU_TUNED_VARIANT}" "flashinfer_jit_cache" "${JIT_VERSION}" "${GPU_TUNED_HW_LABEL}"
        gpu_tuned_verify_build_info "${so}" "flashinfer_jit_cache" "${JIT_VERSION}" "flashinfer_build_info" >/dev/null
    done
    echo "OK: all ${#JIT_SOS[@]} kernel(s) carry a verified build-info stamp"
    rm -f "${JIT_WHEEL}"
    UNPACKED_CONTENT_DIR="$(find "${UNPACK_DIR}" -maxdepth 1 -mindepth 1 -type d)"
    python3 -m wheel pack "${UNPACKED_CONTENT_DIR}" --dest-dir "${REPO_ROOT}/flashinfer-jit-cache/dist"
    rm -rf "${UNPACK_DIR}"
    echo "::endgroup::"

    echo ""
    echo "Build complete."
    echo "To use this build from vllm's venv (no JIT at runtime):"
    echo "  <vllm-venv>/bin/pip install --force-reinstall \\"
    echo "    ${REPO_ROOT}/dist/flashinfer_python-*+${GPU_TUNED_VARIANT}-*.whl \\"
    echo "    ${REPO_ROOT}/flashinfer-jit-cache/dist/flashinfer_jit_cache-*+${GPU_TUNED_VARIANT}-*.whl"
    echo "  # then, in vllm's environment, set FLASHINFER_DISABLE_JIT=1 to turn any"
    echo "  # cache-miss op into a hard error instead of a silent JIT compile."
    echo "  # See tuned/devices/${GPU_TUNED_VARIANT}.conf for known coverage risks --"
    echo "  # run vllm's tuned/verify_fp4_arch_match.py first."
else
    echo ""
    echo "Build complete."
    echo "To use this build from vllm's venv:"
    echo "  <vllm-venv>/bin/pip install --force-reinstall ${REPO_ROOT}/dist/flashinfer_python-*+${GPU_TUNED_VARIANT}-*.whl"
    echo "  # kernels JIT-compile lazily on first call; no flashinfer-jit-cache wheel"
    echo "  # is built for this variant -- see tuned/build.sh's header comment."
fi
