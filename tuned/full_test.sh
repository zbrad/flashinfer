#!/bin/bash
# tuned/full_test.sh <variant> — test the wheels tuned/build.sh made, for ONE variant and ONE CUDA toolkit, and write a
# results log in that toolkit's tree (tuned/build/<cuda-tag>/releases/TEST_RESULTS_<variant>.log).
#
#   CUDA_TAG=cu134 bash tuned/full_test.sh gb10
#
# The unstamped wheels (tuned/build/<cuda-tag>/wheel/) are installed --no-deps into the toolkit's build venv, and every
# test runs from an empty directory with -I, so flashinfer and the jit-cache are the wheels', not the source tree's.
#
# RECORDED DECISION, docs/ARCHITECTURE.md G3 (to be confirmed by the owner): a narrower gate than upstream's pytest
# suite (tests/), which is deliberately not run. The gate is:
#   1. the gates: wheel labels and build commit, the H6 source check, the python and jit-cache wheels are the same
#      version and load from the venv, every shared object in the jit-cache is single-arch (verify_arch) with no PTX,
#      the toolkit libraries resolve (G2);
#   2. tuned/tests/jit_cache_coverage.py: with FLASHINFER_DISABLE_JIT=1 and an empty workspace, trtllm_utils, every XQA
#      shape the cache was built for, and a decode kernel all come from the cache, and nothing is compiled.
# Skips by design, named in the log header: upstream's pytest suite (tests/) and multi-GPU behaviour. The end-to-end model
# check is vllm's tuned/gb10_acceptance_test.sh, run in that repo against these wheels.
#
# The log ends with "=== Full test suite summary: gates + jit-cache coverage passed ===" only when every step passed; release.sh
# greps for that exact line. The other outcome is "=== Full test suite summary: FAILED (see above) ===".
set -uo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/full_test.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

WHEEL_PKGS=(flashinfer_python)
[[ "${GPU_TUNED_NEEDS_AOT_JIT_CACHE}" == "true" ]] && WHEEL_PKGS+=(flashinfer_jit_cache)
WHEELS=()
for pkg in "${WHEEL_PKGS[@]}"; do
    w="$(ls "${GPU_TUNED_WHEEL_DIR}/${pkg}"-*.whl 2>/dev/null | head -1)"
    [[ -n "${w}" ]] || { echo "ERROR: no ${pkg} wheel in ${GPU_TUNED_WHEEL_DIR}. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
    WHEELS+=("${w}")
done
BUILD_COMMIT="$(cat "${GPU_TUNED_BUILD_ROOT}/build-commit" 2>/dev/null || true)"
LABEL="$(cat "${GPU_TUNED_BUILD_ROOT}/release-label" 2>/dev/null || true)"
[[ -n "${BUILD_COMMIT}" && -n "${LABEL}" ]] || { echo "ERROR: ${GPU_TUNED_BUILD_ROOT}/build-commit or release-label missing: the last build did not finish. Re-run tuned/build.sh ${GPU_TUNED_VARIANT}." >&2; exit 1; }
for w in "${WHEELS[@]}"; do
    [[ "$(basename "${w}")" == *"+${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL}-"* ]] \
        || { echo "ERROR: $(basename "${w}") does not carry label ${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL} from release-label." >&2; exit 1; }
done
[[ "${BUILD_COMMIT}" != *-dirty ]] || { echo "ERROR: the build was made from a dirty tree (${BUILD_COMMIT}); commit and rebuild." >&2; exit 1; }
git cat-file -e "${BUILD_COMMIT}^{commit}" 2>/dev/null || { echo "ERROR: build commit ${BUILD_COMMIT} is not in this checkout." >&2; exit 1; }
# H6: the wheels must be from the source at HEAD; only tuned/ may differ.
STALE_SRC="$(git diff --name-only "${BUILD_COMMIT}" HEAD -- . ':!tuned' | head -5)"
if [[ -n "${STALE_SRC}" ]]; then
    echo "ERROR: the wheels are from ${BUILD_COMMIT}, but source outside tuned/ changed since:" >&2
    echo "${STALE_SRC}" | sed 's/^/    /' >&2
    echo "  Rebuild (tuned/build.sh ${GPU_TUNED_VARIANT}) before testing." >&2
    exit 1
fi

PY="${GPU_TUNED_VENV_DIR}/bin/python"
[[ -x "${PY}" ]] || { echo "ERROR: ${PY} not found. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
mkdir -p "${GPU_TUNED_RELEASES_DIR}"
RESULTS_FILE="${GPU_TUNED_RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
rm -f "${RESULTS_FILE}"

# Tuned builds are single-GPU: the tests see exactly one device (GPU_TUNED_TEST_GPUS, default 0).
export CUDA_VISIBLE_DEVICES="${GPU_TUNED_TEST_GPUS:-0}"
TESTDIR="$(mktemp -d)"
trap 'rm -rf "${TESTDIR}"' EXIT

echo "installing the wheels into the build venv (--no-deps)"
"${PY}" -m pip install --quiet --no-deps --force-reinstall "${WHEELS[@]}" \
    || { echo "ERROR: pip could not install the wheels." >&2; exit 1; }

SITE_DIR="$(cd "${TESTDIR}" && "${PY}" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
{
    echo "flashinfer tuned test run -- variant=${GPU_TUNED_VARIANT} cuda=${CUDA_VER} (${CUDA_TAG})"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit:  $(git rev-parse HEAD)"
    echo "build:   ${BUILD_COMMIT} (label ${LABEL})"
    for w in "${WHEELS[@]}"; do echo "wheel:   $(basename "${w}") sha256 $(sha256sum "${w}" | awk '{print $1}')"; done
    echo "torch:   $(cd "${TESTDIR}" && "${PY}" -I -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no device")' 2>&1 | tail -1)"
    echo "scope:   gates; tuned/tests/jit_cache_coverage.py (JIT disabled, empty workspace)"
    echo "devices: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (single-GPU build: multi-GPU tests are not part of its scope)"
    echo "skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests; the model-level check is vllm's acceptance test, run in that repo"
    echo "gates: wheels installed --no-deps in the build venv; python and jit-cache wheels are one version; every jit-cache shared object is single-arch ${GPU_TUNED_FLASHINFER_ARCH} with no PTX; toolkit libs resolve under ${CUDA_HOME}; build commit ${BUILD_COMMIT} == HEAD outside tuned/"
    echo ""
} > "${RESULTS_FILE}"

# Gates. A function, so its status comes back through the tee pipeline's PIPESTATUS.
gate_checks() {
    local rc=0 so n=0 host_only=0 ptx out any
    out="$(cd "${TESTDIR}" && "${PY}" -I -c 'import flashinfer; print(flashinfer.__file__)' 2>&1 | tail -1)"
    if [[ "${out}" == "${SITE_DIR}/"* ]]; then echo "  ok:   flashinfer is the wheel's: ${out}"
    else echo "  FAIL: flashinfer loads from '${out}', not from ${SITE_DIR}"; rc=1; fi
    if [[ "${GPU_TUNED_NEEDS_AOT_JIT_CACHE}" == "true" ]]; then
        any=""
        while IFS= read -r so; do
            n=$((n + 1))
            # A host-only module (no device code) has nothing to check for arch or PTX.
            gpu_tuned_so_has_device_code "${so}" || { host_only=$((host_only + 1)); continue; }
            any="${so}"
            gpu_tuned_verify_arch "${so}" "${GPU_TUNED_FLASHINFER_ARCH}" >/dev/null || { echo "  FAIL: ${so#"${SITE_DIR}"/} is not exactly ${GPU_TUNED_FLASHINFER_ARCH}"; rc=1; }
            # A cuobjdump that cannot run or read the file is a failure, not "no PTX" (H4).
            ptx="$(cuobjdump --list-ptx "${so}" 2>&1)" || { echo "  FAIL: cuobjdump --list-ptx failed on ${so#"${SITE_DIR}"/}: ${ptx}"; rc=1; continue; }
            [[ -z "$(printf '%s\n' "${ptx}" | grep -oE 'sm_[0-9]+[a-z]?' || true)" ]] || { echo "  FAIL: ${so#"${SITE_DIR}"/} embeds PTX"; rc=1; }
        done < <(find "${SITE_DIR}/flashinfer_data" "${SITE_DIR}/flashinfer_jit_cache" -name '*.so' 2>/dev/null | sort)
        [[ ${n} -gt 0 ]] || { echo "  FAIL: no shared objects in the installed jit-cache"; rc=1; }
        echo "  checked ${n} jit-cache shared objects (${host_only} host-only; the rest single-arch ${GPU_TUNED_FLASHINFER_ARCH}, no PTX)"
        if [[ -n "${any}" ]] && gpu_tuned_verify_toolkit_libs "${any}"; then echo "  ok:   toolkit libraries resolve under ${CUDA_HOME}"
        else echo "  FAIL: toolkit libraries do not resolve under ${CUDA_HOME}"; rc=1; fi
    fi
    return "${rc}"
}
# logrun <command...>: run with output to the terminal and the log; returns the command's own status.
logrun() { "$@" 2>&1 | tee -a "${RESULTS_FILE}"; return "${PIPESTATUS[0]}"; }

STATUS=0
echo "=== gates ===" | tee -a "${RESULTS_FILE}"
logrun gate_checks || STATUS=1

if [[ "${GPU_TUNED_NEEDS_AOT_JIT_CACHE}" == "true" ]]; then
    echo "" | tee -a "${RESULTS_FILE}"
    echo "=== tuned/tests/jit_cache_coverage.py (FLASHINFER_DISABLE_JIT=1, empty workspace) ===" | tee -a "${RESULTS_FILE}"
    COVERAGE_WS="$(mktemp -d)"
    (cd "${TESTDIR}" && env FLASHINFER_DISABLE_JIT=1 FLASHINFER_WORKSPACE_BASE="${COVERAGE_WS}" \
        FLASHINFER_AOT_XQA_INPUT_TYPE="${GPU_TUNED_AOT_XQA_INPUT_TYPE:-}" FLASHINFER_AOT_XQA_FP8_KV_CACHE="${GPU_TUNED_AOT_XQA_FP8_KV_CACHE:-}" \
        FLASHINFER_AOT_XQA_TOKEN_PER_PAGE="${GPU_TUNED_AOT_XQA_TOKEN_PER_PAGE:-}" FLASHINFER_AOT_XQA_HEAD_SIZE="${GPU_TUNED_AOT_XQA_HEAD_SIZE:-}" \
        FLASHINFER_AOT_XQA_HEAD_GRP_SIZE="${GPU_TUNED_AOT_XQA_HEAD_GRP_SIZE:-}" \
        "${PY}" -I "${REPO_ROOT}/tuned/tests/jit_cache_coverage.py") 2>&1 | tee -a "${RESULTS_FILE}"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { STATUS=1; echo "jit_cache_coverage.py FAILED" | tee -a "${RESULTS_FILE}"; }
    rm -rf "${COVERAGE_WS}"
fi

echo "" >> "${RESULTS_FILE}"
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RESULTS_FILE}"
if [[ "${STATUS}" -eq 0 ]]; then
    echo "=== Full test suite summary: gates + jit-cache coverage passed ===" | tee -a "${RESULTS_FILE}"
else
    echo "=== Full test suite summary: FAILED (see above) ===" | tee -a "${RESULTS_FILE}"
fi
echo "Results written to ${RESULTS_FILE}"
exit "${STATUS}"
