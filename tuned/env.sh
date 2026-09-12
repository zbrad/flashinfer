#!/bin/bash
# tuned/env.sh <variant> — device config + GPU-match assertion for a tuned
# single-arch flashinfer build (gb10/rtx40/rtx50). Source this file with
# the variant as $1; do not execute it directly.
#
# Exported: GPU_TUNED_VARIANT, GPU_TUNED_COMPUTE_CAP, GPU_TUNED_CUDA_ARCH,
# GPU_TUNED_HW_LABEL (from tuned/devices/<variant>.conf), plus
# FLASHINFER_CUDA_ARCH_LIST (set to GPU_TUNED_CUDA_ARCH, the single value
# this whole tuned build pins compilation to instead of the full multi-arch
# release matrix).
#
# Note: unlike some of the other tuned-build repos, this doesn't assert a
# CPU/host platform (uname -m) -- flashinfer's build is gated purely by
# GPU compute capability (via nvidia-smi below), matching this repo's own
# pre-existing convention (the original scripts/build_gb10.sh never checked
# uname -m either, even though GB10 implies an aarch64 Grace CPU host).

GPU_TUNED_ARG_VARIANT="$1"
if [[ -z "${GPU_TUNED_ARG_VARIANT}" ]]; then
    echo "ERROR: env.sh requires a variant argument (gb10/rtx40/rtx50)" >&2
    return 1 2>/dev/null || exit 1
fi

GPU_TUNED_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=devices/rtx50.conf
source "${GPU_TUNED_SELF_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" || return 1 2>/dev/null || exit 1
# shellcheck source=common.sh
# Vendored from https://github.com/zbrad/tuned-common (pinned commit --
# see common.sh's own header/sync instructions to update). Provides
# gpu_tuned_assert_compute_cap, shared verbatim across the fleet instead
# of hand-copied-and-edited per repo.
source "${GPU_TUNED_SELF_DIR}/common.sh" || return 1 2>/dev/null || exit 1

# Fail loudly if this isn't the intended GPU, rather than silently building
# for whatever's actually present -- same check every one of the original
# scripts/build_*.sh had individually, now shared.
gpu_tuned_assert_compute_cap "${GPU_TUNED_COMPUTE_CAP}" "${GPU_TUNED_HW_LABEL}" || return 1 2>/dev/null || exit 1

export FLASHINFER_CUDA_ARCH_LIST="${GPU_TUNED_CUDA_ARCH}"

# embed_build_info <so_path> <variant> <package> <version> [hw_label] —
# thin wrapper over gpu_tuned_embed_build_info (common.sh) that pins the
# section name to .flashinfer_build_info, same convention as every other
# tuned-builds repo's own embed_build_info wrapper (each pins its own
# name rather than relying on the base function's package-derived
# default, so a caller passing a different <package> string per call --
# e.g. this repo's own "flashinfer_jit_cache" -- still lands in one
# predictable, greppable section name).
embed_build_info() {
    local so_path="$1" variant="$2" package="$3" version="$4" hw_label="$5"
    gpu_tuned_embed_build_info "${so_path}" "${variant}" "${package}" "${version}" \
        "${hw_label}" "https://github.com/zbrad/flashinfer" "flashinfer_build_info"
}
