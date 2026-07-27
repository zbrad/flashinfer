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
export GPU_TUNED_VARIANT GPU_TUNED_COMPUTE_CAP GPU_TUNED_CUDA_ARCH GPU_TUNED_HW_LABEL

# Fail loudly if this isn't the intended GPU, rather than silently building
# for whatever's actually present -- same check every one of the original
# scripts/build_*.sh had individually, now shared.
GPU_TUNED_DETECTED_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')"
if [[ "${GPU_TUNED_DETECTED_CC}" != "${GPU_TUNED_COMPUTE_CAP}" ]]; then
    echo "ERROR: tuned/env.sh: detected GPU compute capability '${GPU_TUNED_DETECTED_CC}'," \
         "expected '${GPU_TUNED_COMPUTE_CAP}' (${GPU_TUNED_HW_LABEL})." >&2
    echo "       This is the tuned-builds branch; use upstream main for other GPUs." >&2
    return 1 2>/dev/null || exit 1
fi

export FLASHINFER_CUDA_ARCH_LIST="${GPU_TUNED_CUDA_ARCH}"
