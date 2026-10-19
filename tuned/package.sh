#!/bin/bash
# tuned/package.sh <variant> — package the wheels tuned/build.sh made and tuned/full_test.sh tested: stamp a STAGED copy
# of the flashinfer-jit-cache wheel with the JSON build info on every shared object that has device code (docs/ARCHITECTURE.md
# H7), repack, copy the python wheel beside it (it holds no compiled code, so there is nothing to stamp), write a sha256
# sidecar for each, package-info.txt and, if no notes are tracked yet, a DRAFT release-notes file. It never builds, tests,
# tags or publishes (B3). Stage order:
#   build.sh -> full_test.sh -> package.sh -> smoke_test.sh -> (review and commit the notes) -> release.sh -> publish.sh
#
#   CUDA_TAG=cu134 bash tuned/package.sh gb10
#
# Input:  tuned/build/<cuda-tag>/wheel/flashinfer_{python,jit_cache}-*.whl (unstamped; never modified here, H3),
#         release-label and build-commit from build.sh, and the passing test log from full_test.sh.
# Output: tuned/build/<cuda-tag>/dist/{*.whl, *.whl.sha256, package-info.txt} and tuned/releases/RELEASE_NOTES_<version>.md
#         (DRAFT) when no notes exist. package-info.txt is written last (H2): its presence means every check passed.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/package.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

[[ -d "${GPU_TUNED_VENV_DIR}" ]] || { echo "ERROR: ${GPU_TUNED_VENV_DIR} not found. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
gpu_tuned_verify_venv "${GPU_TUNED_VENV_DIR}" "${REPO_ROOT}"
# shellcheck source=/dev/null
source "${GPU_TUNED_VENV_DIR}/bin/activate"

WHEEL_PKGS=(flashinfer_python)
[[ "${GPU_TUNED_NEEDS_AOT_JIT_CACHE}" == "true" ]] && WHEEL_PKGS+=(flashinfer_jit_cache)
BUILT_WHEELS=()
for pkg in "${WHEEL_PKGS[@]}"; do
    w="$(ls "${GPU_TUNED_WHEEL_DIR}/${pkg}"-*.whl 2>/dev/null | head -1)"
    [[ -n "${w}" ]] || { echo "ERROR: no ${pkg} wheel in ${GPU_TUNED_WHEEL_DIR}. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
    BUILT_WHEELS+=("${w}")
done
LABEL="$(cat "${GPU_TUNED_BUILD_ROOT}/release-label" 2>/dev/null || true)"
BUILD_COMMIT="$(cat "${GPU_TUNED_BUILD_ROOT}/build-commit" 2>/dev/null || true)"
[[ -n "${LABEL}" && -n "${BUILD_COMMIT}" ]] || { echo "ERROR: release-label or build-commit missing in ${GPU_TUNED_BUILD_ROOT}: the last build did not finish." >&2; exit 1; }
[[ "${BUILD_COMMIT}" != *-dirty ]] || { echo "ERROR: the build was made from a dirty tree (${BUILD_COMMIT}); commit and rebuild." >&2; exit 1; }

# The label is the one build.sh allocated and baked in; it is never recomputed here (E4).
VERSION="$(gpu_tuned_wheel_version "${BUILT_WHEELS[0]}" flashinfer_python)"
for i in "${!BUILT_WHEELS[@]}"; do
    v="$(gpu_tuned_wheel_version "${BUILT_WHEELS[$i]}" "${WHEEL_PKGS[$i]}")"
    [[ "${v}" == "${VERSION}" && "${v}" == *"+${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL}" ]] || {
        echo "ERROR: $(basename "${BUILT_WHEELS[$i]}") is version ${v}; every wheel must be ${VERSION}, carrying +${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL} from release-label." >&2
        exit 1
    }
    gpu_tuned_check_wheel_label "${BUILT_WHEELS[$i]}"
done
BASE_VERSION="${VERSION%%+*}"
RELEASE_TAG="v${VERSION}"
gpu_tuned_check_release_tag "${RELEASE_TAG}"

# H6: refuse a build from other source than HEAD; HEAD may differ from the build only in tuned/.
git cat-file -e "${BUILD_COMMIT}^{commit}" 2>/dev/null || { echo "ERROR: build commit ${BUILD_COMMIT} is not in this checkout." >&2; exit 1; }
STALE_SRC="$(git diff --name-only "${BUILD_COMMIT}" HEAD -- . ':!tuned' | head -5)"
if [[ -n "${STALE_SRC}" ]]; then
    echo "ERROR: the wheels are from ${BUILD_COMMIT}, but source outside tuned/ changed since:" >&2
    echo "${STALE_SRC}" | sed 's/^/    /' >&2
    echo "  Rebuild and re-test before packaging." >&2
    exit 1
fi
TEST_LOG="${GPU_TUNED_RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
[[ -f "${TEST_LOG}" ]] || { echo "ERROR: ${TEST_LOG} not found; run CUDA_TAG=${CUDA_TAG} bash tuned/full_test.sh ${GPU_TUNED_VARIANT} before packaging." >&2; exit 1; }
grep -q "^=== Full test suite summary: gates + jit-cache coverage passed ===\$" "${TEST_LOG}" \
    || { echo "ERROR: ${TEST_LOG} does not show a full, passing run." >&2; exit 1; }
for w in "${BUILT_WHEELS[@]}"; do
    [[ "${w}" -nt "${TEST_LOG}" ]] && { echo "ERROR: $(basename "${w}") is newer than the test log; re-run tuned/full_test.sh ${GPU_TUNED_VARIANT}." >&2; exit 1; }
    tested="$(sed -n "s/^wheel: *$(basename "${w}") sha256 //p" "${TEST_LOG}" | head -1)"
    [[ "${tested}" == "$(sha256sum "${w}" | awk '{print $1}')" ]] || { echo "ERROR: the test log tested a different $(basename "${w}") (sha256 ${tested:-none})." >&2; exit 1; }
done
TESTED_COMMIT="$(sed -n 's/^commit: *//p' "${TEST_LOG}" | head -1)"

echo "=========================================="
echo "Packaging flashinfer wheels (${GPU_TUNED_HW_LABEL})"
echo "=========================================="
echo "Version:      ${VERSION}"
echo "Release tag: ${RELEASE_TAG}"
echo "Built at:     ${BUILD_COMMIT}; tested at ${TESTED_COMMIT}; packaging at $(git rev-parse HEAD)"
echo ""

mkdir -p "${GPU_TUNED_DIST_DIR}"
# package-info.txt is removed first and written last: its presence means every check below passed.
rm -f "${GPU_TUNED_DIST_DIR}/package-info.txt" "${GPU_TUNED_DIST_DIR}"/flashinfer_*.whl "${GPU_TUNED_DIST_DIR}"/flashinfer_*.whl.sha256

command -v readelf >/dev/null && command -v objcopy >/dev/null && command -v cuobjdump >/dev/null \
    || { echo "ERROR: readelf, objcopy and cuobjdump are needed for the packaging checks (H4)." >&2; exit 1; }

UNPACK_DIR="$(mktemp -d)"
trap 'rm -rf "${UNPACK_DIR}"' EXIT
INFO_LINES=()
for i in "${!BUILT_WHEELS[@]}"; do
    BUILT="${BUILT_WHEELS[$i]}"; PKG="${WHEEL_PKGS[$i]}"; BUILT_NAME="$(basename "${BUILT}")"
    if [[ "${PKG}" == flashinfer_jit_cache ]]; then
        echo "Stamping build info into a staged copy of ${BUILT_NAME}"
        STAGE="${UNPACK_DIR}/${PKG}"; mkdir -p "${STAGE}"
        python3 -m wheel unpack "${BUILT}" --dest "${STAGE}"
        UNPACKED="$(find "${STAGE}" -maxdepth 1 -mindepth 1 -type d)"
        [[ -d "${UNPACKED}" ]] || { echo "ERROR: ${BUILT_NAME} did not unpack to one directory." >&2; exit 1; }
        stamped=0; host_only=0
        while IFS= read -r so; do
            if ! gpu_tuned_so_has_device_code "${so}"; then
                # A host-only module has no device code to describe; it is listed, not stamped.
                host_only=$((host_only + 1)); continue
            fi
            gpu_tuned_verify_arch "${so}" "${GPU_TUNED_FLASHINFER_ARCH}" >/dev/null || { echo "ERROR: ${so#"${STAGE}"/} is not exactly ${GPU_TUNED_FLASHINFER_ARCH}." >&2; exit 1; }
            # Pre-compiled only: PTX would be JIT-compiled by the driver at first use. A cuobjdump that cannot run or read
            # the file is a failure, not "no PTX" (H4).
            PTX_OUT="$(cuobjdump --list-ptx "${so}" 2>&1)" \
                || { echo "ERROR: cuobjdump --list-ptx failed on ${so#"${STAGE}"/}; the PTX check cannot run: ${PTX_OUT}" >&2; exit 1; }
            PTX_FOUND="$(printf '%s\n' "${PTX_OUT}" | grep -oE 'sm_[0-9]+[a-z]?' | sort -u | tr '\n' ' ' || true)"
            [[ -z "${PTX_FOUND// /}" ]] || { echo "ERROR: ${so#"${STAGE}"/} embeds PTX (${PTX_FOUND}): the driver would JIT-compile it at runtime." >&2; exit 1; }
            # fork.commit is the commit the wheel was built from, not HEAD at packaging (package-info.txt has both).
            gpu_tuned_stamp_lib_json "${so}" "${PKG}" "${VERSION}" "" \
                "upstream.declared_version=${BASE_VERSION}" "fork.commit=${BUILD_COMMIT}" "fork.dirty:bool=false" \
                || { echo "ERROR: could not stamp ${so#"${STAGE}"/}." >&2; exit 1; }
            stamped=$((stamped + 1))
        done < <(find "${UNPACKED}" -name '*.so' | sort)
        [[ ${stamped} -gt 0 ]] || { echo "ERROR: no shared object with device code in ${BUILT_NAME}; nothing to stamp." >&2; exit 1; }
        echo "OK: ${stamped} shared objects stamped and read back; ${host_only} host-only left unstamped"
        python3 -m wheel pack "${UNPACKED}" --dest-dir "${GPU_TUNED_DIST_DIR}"
        INFO_LINES+=("stamped_objects=${stamped}" "host_only_objects=${host_only}")
    else
        # Pure Python (py3-none-any): no compiled code, nothing to stamp or to check for PTX. Copied byte for byte.
        cp -p "${BUILT}" "${GPU_TUNED_DIST_DIR}/${BUILT_NAME}"
    fi
    WHEEL="${GPU_TUNED_DIST_DIR}/${BUILT_NAME}"
    [[ -f "${WHEEL}" ]] || { echo "ERROR: packaged wheel is not ${BUILT_NAME}." >&2; exit 1; }
    gpu_tuned_check_wheel_label "${WHEEL}"
    gpu_tuned_write_sha256_sidecar "${WHEEL}"
    SHA="$(awk '{print $1}' "${WHEEL}.sha256")"
    echo "Wheel:  ${BUILT_NAME} ($(du -sh "${WHEEL}" | awk '{print $1}'))  sha256 ${SHA}"
    INFO_LINES+=("wheel_${PKG#flashinfer_}=${BUILT_NAME}" "sha256_${PKG#flashinfer_}=${SHA}")
done

# H5: a DRAFT notes file if none is tracked yet; release.sh refuses it until a person has reviewed it and removed the
# banner. The numbers come from the attached test log (G4).
NOTES="tuned/releases/RELEASE_NOTES_${RELEASE_TAG#v}.md"
if [[ ! -f "${NOTES}" ]]; then
    mkdir -p tuned/releases
    TAG_URL="$(echo "${RELEASE_TAG}" | sed 's/+/%2B/g')"
    {
        echo "<!-- DRAFT — not yet published: review every line, then delete this banner and commit. -->"
        echo "# flashinfer ${BASE_VERSION} — ${GPU_TUNED_VARIANT} ${CUDA_TAG} (${RELEASE_TAG#v})"
        echo ""
        echo "Single-arch flashinfer wheels for ${GPU_TUNED_HW_LABEL} (FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}), CUDA ${CUDA_VER}."
        echo ""
        echo "- Release label: ${LABEL} (UTC date of the tuned commit)"
        echo "- Built from commit ${BUILD_COMMIT}; tested at ${TESTED_COMMIT}"
        echo "- Upstream base: ${GPU_TUNED_UPSTREAM_REF} merge-base $(git merge-base HEAD "${GPU_TUNED_UPSTREAM_REF}" 2>/dev/null || echo unknown)"
        for line in "${INFO_LINES[@]}"; do case "${line}" in sha256_*) echo "- ${line%%=*}: ${line#*=}" ;; esac; done
        echo ""
        echo "## Tests"
        echo ""
        echo "Scope (from the attached test log): $(sed -n 's/^scope: *//p' "${TEST_LOG}" | head -1)"
        echo "Skips by design: $(sed -n 's/^skips by design: *//p' "${TEST_LOG}" | head -1)"
        echo "Result: $(grep '^=== Full test suite summary' "${TEST_LOG}" | tail -1)"
        echo ""
        echo "## Install"
        echo ""
        echo '```'
        for w in "${BUILT_WHEELS[@]}"; do
            echo "pip install --no-deps \"https://github.com/zbrad/flashinfer/releases/download/${TAG_URL}/$(basename "${w}" | sed 's/+/%2B/g')\""
        done
        echo '```'
    } > "${NOTES}"
    echo "Wrote DRAFT release notes: ${NOTES}"
fi

# Written last.
{
    echo "release_tag=${RELEASE_TAG}"
    echo "version=${VERSION}"
    echo "upstream_version=${BASE_VERSION}"
    echo "label=${LABEL}"
    echo "commit=$(git rev-parse HEAD)"
    echo "built_commit=${BUILD_COMMIT}"
    echo "tested_commit=${TESTED_COMMIT}"
    echo "cuda_tag=${CUDA_TAG}"
    echo "variant=${GPU_TUNED_VARIANT}"
    printf '%s\n' "${INFO_LINES[@]}"
} > "${GPU_TUNED_DIST_DIR}/package-info.txt"

echo ""
echo "Packaged (not published): ${RELEASE_TAG}"
echo "Next: CUDA_TAG=${CUDA_TAG} bash tuned/smoke_test.sh ${GPU_TUNED_VARIANT}, review and commit ${NOTES}, then CUDA_TAG=${CUDA_TAG} bash tuned/release.sh ${GPU_TUNED_VARIANT}"
