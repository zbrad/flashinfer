#!/bin/bash
# tuned/smoke_test.sh <variant> — prove the PACKAGED wheels work, before they are released. It installs the stamped wheels
# tuned/package.sh made (named in package-info.txt) into the toolkit's build venv with --no-deps, and checks:
#   1. each wheel's sha256 matches package-info.txt and its .sha256 sidecar;
#   2. every shared object installed from the jit-cache is byte-identical to the packaged wheel's copy, so the bits in
#      the package are the bits that run;
#   3. the installed distributions are exactly the packaged version, flashinfer loads from the venv, and the build-info
#      JSON of every shared object with device code reads back as this release, variant and toolkit (H7);
#   4. the toolkit libraries resolve (G2);
#   5. tuned/tests/jit_cache_coverage.py passes against the package: trtllm_utils, every configured XQA shape and a
#      decode kernel come from the cache with FLASHINFER_DISABLE_JIT=1 and an empty workspace, and nothing compiles.
# A smoke test, not the suite: tuned/full_test.sh ran the gate against the unstamped wheels. Writes SMOKE_RESULTS_<variant>.log
# beside TEST_RESULTS; release.sh requires a passing log newer than the packaged wheels. A pass needs the end marker, so
# an abort cannot pass.
set -uo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/smoke_test.sh <variant>}"
REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPODIR}" || exit 1
# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

INFO="${GPU_TUNED_DIST_DIR}/package-info.txt"
[[ -f "${INFO}" ]] || { echo "ERROR: ${INFO} not found. Run CUDA_TAG=${CUDA_TAG} bash tuned/package.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
info() { sed -n "s/^$1=//p" "${INFO}"; }
VERSION="$(info version)"
KINDS=(python); [[ "${GPU_TUNED_NEEDS_AOT_JIT_CACHE}" == "true" ]] && KINDS+=(jit_cache)
WHEELS=()
for k in "${KINDS[@]}"; do
    w="${GPU_TUNED_DIST_DIR}/$(info "wheel_${k}")"
    [[ -f "${w}" ]] || { echo "ERROR: ${w} not found." >&2; exit 1; }
    WHEELS+=("${w}")
done
mkdir -p "${GPU_TUNED_RELEASES_DIR}"
LOG="${GPU_TUNED_RELEASES_DIR}/SMOKE_RESULTS_${GPU_TUNED_VARIANT}.log"
rm -f "${LOG}"

PY="${GPU_TUNED_VENV_DIR}/bin/python"
[[ -x "${PY}" ]] || { echo "ERROR: ${PY} not found. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
FAILS=0
fail() { echo "  FAIL: $*"; FAILS=$((FAILS + 1)); }
ok() { echo "  ok:   $*"; }

{
    echo "flashinfer tuned smoke test -- variant=${GPU_TUNED_VARIANT} cuda=${CUDA_VER} (${CUDA_TAG})"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for w in "${WHEELS[@]}"; do echo "wheel:   $(basename "${w}")"; done
    echo "packaged commit: $(info commit)"
    echo ""

    for i in "${!KINDS[@]}"; do
        w="${WHEELS[$i]}"; k="${KINDS[$i]}"
        got="$(sha256sum "${w}" | awk '{print $1}')"
        [[ "${got}" == "$(info "sha256_${k}")" ]] && ok "$(basename "${w}") sha256 matches package-info.txt" || fail "$(basename "${w}") sha256 ${got} does not match package-info.txt"
        gpu_tuned_verify_sha256_sidecar "${w}" && ok "$(basename "${w}") sidecar .sha256 verifies" || fail "$(basename "${w}") sidecar .sha256 does not verify"
    done

    # Install the packaged wheels over the unstamped ones the test run used. --no-deps: the venv holds the requirements.
    if "${PY}" -m pip install --quiet --no-deps --force-reinstall "${WHEELS[@]}"; then
        ok "packaged wheels installed into ${GPU_TUNED_VENV_DIR} (--no-deps)"
    else
        fail "pip could not install the packaged wheels"
    fi
    SITE_DIR="$(cd "${WORK}" && "${PY}" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

    meta="$(cd "${WORK}" && "${PY}" -I -c 'import importlib.metadata as md, flashinfer; print("python=" + md.version("flashinfer-python")); print("cache=" + md.version("flashinfer-jit-cache")); print("file=" + flashinfer.__file__)' 2>&1)"
    echo "${meta}" | sed 's/^/  | /'
    [[ "$(sed -n 's/^python=//p' <<< "${meta}")" == "${VERSION}" ]] && ok "installed flashinfer-python is exactly ${VERSION}" || fail "installed flashinfer-python is not ${VERSION}"
    if [[ "${GPU_TUNED_NEEDS_AOT_JIT_CACHE}" == "true" ]]; then
        [[ "$(sed -n 's/^cache=//p' <<< "${meta}")" == "${VERSION}" ]] && ok "installed flashinfer-jit-cache is exactly ${VERSION}" || fail "installed flashinfer-jit-cache is not ${VERSION}"
    fi
    [[ "$(sed -n 's/^file=//p' <<< "${meta}")" == "${SITE_DIR}/"* ]] && ok "flashinfer loads from the venv" || fail "flashinfer loads outside ${SITE_DIR}"

    if [[ "${GPU_TUNED_NEEDS_AOT_JIT_CACHE}" == "true" ]]; then
        CACHE_WHEEL="${WHEELS[1]}"
        # Byte identity of every installed shared object with the packaged wheel's copy.
        diffs="$("${PY}" -I - "${CACHE_WHEEL}" "${SITE_DIR}" <<'PY'
import hashlib, os, sys, zipfile
wheel, site = sys.argv[1:3]
bad, n = [], 0
with zipfile.ZipFile(wheel) as z:
    for name in z.namelist():
        if not name.endswith(".so"):
            continue
        n += 1
        # Files under <dist>.data/purelib/ install into site-packages itself.
        path = os.path.join(site, name.split(".data/purelib/", 1)[1] if ".data/purelib/" in name else name)
        if not os.path.isfile(path) or hashlib.sha256(open(path, "rb").read()).hexdigest() != hashlib.sha256(z.read(name)).hexdigest():
            bad.append(name)
print(f"{n} {len(bad)} " + " ".join(bad[:5]))
PY
)"
        read -r n_so n_bad rest <<< "${diffs}"
        [[ "${n_so}" -gt 0 && "${n_bad}" -eq 0 ]] && ok "all ${n_so} installed jit-cache shared objects are byte-identical to the packaged wheel's copies" \
            || fail "${n_bad} of ${n_so} installed shared objects differ from the packaged wheel: ${rest}"

        stamped=0; unstamped=0; badstamp=0
        while IFS= read -r so; do
            gpu_tuned_so_has_device_code "${so}" || continue
            if gpu_tuned_verify_build_info_json "${so}" "release=${VERSION}" "variant=${GPU_TUNED_VARIANT}" "cuda.tag=${CUDA_TAG}" >/dev/null 2>&1 \
                && gpu_tuned_verify_build_info_json "${so}" "fork.commit=$(info built_commit)" >/dev/null 2>&1; then
                stamped=$((stamped + 1))
            else
                badstamp=$((badstamp + 1))
            fi
        done < <(find "${SITE_DIR}/flashinfer_jit_cache" -name '*.so' | sort)
        [[ ${stamped} -eq "$(info stamped_objects)" && ${badstamp} -eq 0 ]] \
            && ok "${stamped} shared objects: build-info stamp reads back as release ${VERSION}, ${GPU_TUNED_VARIANT}, ${CUDA_TAG} and the build commit" \
            || fail "stamps: ${stamped} read back (packaged: $(info stamped_objects)), ${badstamp} wrong or missing"

        first_so="$(find "${SITE_DIR}/flashinfer_jit_cache" -name '*.so' | sort | while IFS= read -r f; do gpu_tuned_so_has_device_code "${f}" && { echo "${f}"; break; }; done)"
        gpu_tuned_verify_toolkit_libs "${first_so}" && ok "toolkit libraries resolve under ${CUDA_HOME}" || fail "toolkit libraries do not resolve under ${CUDA_HOME}"

        # The packaged pair against the configured coverage, JIT disabled, empty workspace.
        out="$(cd "${WORK}" && env FLASHINFER_DISABLE_JIT=1 FLASHINFER_WORKSPACE_BASE="${WORK}/ws" \
            FLASHINFER_AOT_XQA_INPUT_TYPE="${GPU_TUNED_AOT_XQA_INPUT_TYPE:-}" FLASHINFER_AOT_XQA_FP8_KV_CACHE="${GPU_TUNED_AOT_XQA_FP8_KV_CACHE:-}" \
            FLASHINFER_AOT_XQA_TOKEN_PER_PAGE="${GPU_TUNED_AOT_XQA_TOKEN_PER_PAGE:-}" FLASHINFER_AOT_XQA_HEAD_SIZE="${GPU_TUNED_AOT_XQA_HEAD_SIZE:-}" \
            FLASHINFER_AOT_XQA_HEAD_GRP_SIZE="${GPU_TUNED_AOT_XQA_HEAD_GRP_SIZE:-}" \
            "${PY}" -I "${REPODIR}/tuned/tests/jit_cache_coverage.py" 2>&1)"; rc=$?
        printf '%s\n' "${out}" | tail -9 | sed 's/^/  | /'
        [[ ${rc} -eq 0 ]] && ok "jit_cache_coverage passes against the package (JIT disabled, empty workspace)" || fail "jit_cache_coverage (exit ${rc})"
    fi

    echo ""
    if [[ ${FAILS} -eq 0 ]]; then
        echo "=== Smoke test summary: passed ==="
    else
        echo "=== Smoke test summary: FAILED (${FAILS}) ==="
    fi
    echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "${LOG}"
grep -q '^=== Smoke test summary: passed ===$' "${LOG}"
