#!/bin/bash
# tuned/tests/extra_tests.sh <variant> — checks run before a release, beyond tuned/full_test.sh. Not part of the release gates;
# the numbers it prints are what the release notes quote. Needs the packaged wheels (tuned/package.sh) and uses each venv's own python.
#
#   CUDA_TAG=cu134 bash tuned/tests/extra_tests.sh gb10 acceptance   # vLLM's acceptance test against these wheels, JIT disabled
#   CUDA_TAG=cu134 bash tuned/tests/extra_tests.sh gb10 flashinfer    # a slice of flashinfer's own tests/
#
# acceptance: installs the packaged wheels (--no-deps) into the vllm repo's toolkit venv (VLLM_REPO, default ../vllm; its
#   tuned/build/<cuda-tag>/venv must exist), then runs vllm's tuned/gb10_acceptance_test.sh with FLASHINFER_DISABLE_JIT=1 and an empty
#   FLASHINFER_WORKSPACE_BASE, at ACCEPT_GPU_MEM_UTIL=0.25 (0.85 hung the GPU driver and rebooted the node, 2026-10-19). Passes only if
#   both NemotronH stand-ins generate and no shared object was compiled into the workspace.
# flashinfer: tests/attention/test_xqa_batch_decode.py::test_xqa_batch_decode, only the ids for the shapes in tuned/repo.env (q bf16,
#   kv bf16 or fp8, o bf16, pages 16/32/64, head group 8/16, head_dim 128), in this repo's build venv; and all of
#   tests/mamba/test_selective_state_update_stp.py in the vllm venv (it needs triton). JIT is allowed here: these tests exercise many
#   shapes beyond the cache, so they check the kernels, not the cache; the shared objects compiled during the run are listed.
set -uo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/tests/extra_tests.sh <variant> acceptance|flashinfer}"
WHICH="${2:?usage: tuned/tests/extra_tests.sh <variant> acceptance|flashinfer}"
REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1
VLLM_REPO="${VLLM_REPO:-${REPODIR}/../vllm}"
VLLM_PY="${VLLM_REPO}/tuned/build/${CUDA_TAG}/venv/bin/python"
FI_PY="${GPU_TUNED_VENV_DIR}/bin/python"
INFO="${GPU_TUNED_DIST_DIR}/package-info.txt"
[[ -f "${INFO}" ]] || { echo "ERROR: ${INFO} not found; run tuned/package.sh first." >&2; exit 1; }
[[ -x "${VLLM_PY}" ]] || { echo "ERROR: ${VLLM_PY} not found; build the vllm venv first (VLLM_REPO=${VLLM_REPO})." >&2; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
COMPILED() { find "$1" -name '*.so' 2>/dev/null | sed 's|.*/cached_ops/||; s|/.*||'; }
export CUDA_VISIBLE_DEVICES="${GPU_TUNED_TEST_GPUS:-0}"

# run_pytest <python> <workspace> <pytest args...>: from an empty directory, -I, the repo appended AFTER site-packages so the
# installed wheels are imported and the repo supplies only tests/.
run_pytest() {
    local py="$1" ws="$2"; shift 2
    (cd "${WORK}" && env FLASHINFER_WORKSPACE_BASE="${ws}" MAX_JOBS="${MAX_JOBS:-6}" "${py}" -I -c '
import sys
sys.path.append(sys.argv[1])
import pytest
raise SystemExit(pytest.main(sys.argv[2:]))
' "${REPODIR}" -q -p no:cacheprovider --import-mode=importlib --maxfail=20 -rs "$@")
}

install_wheels() {
    local py="$1" w
    local wheels=()
    for k in python jit_cache; do wheels+=("${GPU_TUNED_DIST_DIR}/$(sed -n "s/^wheel_${k}=//p" "${INFO}")"); done
    for w in "${wheels[@]}"; do [[ -f "${w}" ]] || { echo "ERROR: ${w} not found." >&2; exit 1; }; done
    "${py}" -m pip install --quiet --no-deps --force-reinstall "${wheels[@]}" || exit 1
}

case "${WHICH}" in
acceptance)
    install_wheels "${VLLM_PY}"
    WS="${WORK}/ws"; mkdir -p "${WS}"
    (cd "${VLLM_REPO}" && env FLASHINFER_DISABLE_JIT=1 FLASHINFER_WORKSPACE_BASE="${WS}" ACCEPT_GPU_MEM_UTIL="${ACCEPT_GPU_MEM_UTIL:-0.25}" \
        TUNED_PYTHON="${VLLM_PY}" bash tuned/gb10_acceptance_test.sh) 2>&1 | tee "${WORK}/acceptance.log"
    rc=${PIPESTATUS[0]}
    n="$(COMPILED "${WS}" | wc -l)"
    echo "shared objects compiled into the empty workspace: ${n}"
    [[ ${rc} -eq 0 && ${n} -eq 0 ]] && grep -q '^PASS  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4' "${WORK}/acceptance.log" \
        && grep -q '^PASS  nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16' "${WORK}/acceptance.log" \
        && echo "=== acceptance: passed ===" || { echo "=== acceptance: FAILED ==="; exit 1; }
    ;;
flashinfer)
    install_wheels "${FI_PY}"; install_wheels "${VLLM_PY}"
    "${FI_PY}" -m pip install --quiet pytest || exit 1
    WS1="${WORK}/ws-xqa"; WS2="${WORK}/ws-mamba"; mkdir -p "${WS1}" "${WS2}"
    # Collect the ids, keep the shapes this release ships.
    run_pytest "${FI_PY}" "${WS1}" --collect-only "${REPODIR}/tests/attention/test_xqa_batch_decode.py::test_xqa_batch_decode" \
        | grep '^tests/' | "${FI_PY}" -I -c '
import re, sys
pat = re.compile(r"-bf16-(bf16|fp8)-bf16--?\d+-\d+-\d+-(16|32|64)-\d+-(8|16)-128\]$")
for line in sys.stdin:
    line = line.strip()
    if pat.search(line):
        print(sys.argv[1] + "/" + line)
' "${REPODIR}" > "${WORK}/ids.txt"
    echo "selected $(wc -l < "${WORK}/ids.txt") xqa batch decode tests"
    [[ -s "${WORK}/ids.txt" ]] || { echo "ERROR: no test ids selected." >&2; exit 1; }
    FAILS=0
    mapfile -t IDS < "${WORK}/ids.txt"
    run_pytest "${FI_PY}" "${WS1}" "${IDS[@]}" || FAILS=$((FAILS + 1))
    echo "modules compiled during the xqa run (not in the cache):"; COMPILED "${WS1}" | sed 's/^/  /'
    run_pytest "${VLLM_PY}" "${WS2}" "${REPODIR}/tests/mamba/test_selective_state_update_stp.py" || FAILS=$((FAILS + 1))
    echo "modules compiled during the mamba run: $(COMPILED "${WS2}" | wc -l)"
    [[ ${FAILS} -eq 0 ]] && echo "=== flashinfer slice: passed ===" || { echo "=== flashinfer slice: FAILED ==="; exit 1; }
    ;;
*) echo "usage: tuned/tests/extra_tests.sh <variant> acceptance|flashinfer" >&2; exit 2 ;;
esac
