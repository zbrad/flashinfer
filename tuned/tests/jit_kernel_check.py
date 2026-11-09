"""JIT-compile and run flashinfer attention kernels on the GPU and compare them with a float32 torch reference.

For a variant that ships no AOT jit-cache (rtx40), the kernels compile on first call, so this is the check that
they compile for the device and compute the right thing. Run from an empty directory against the installed wheel:

    FLASHINFER_WORKSPACE_BASE=<empty dir> python -I jit_kernel_check.py

Checks single decode, causal single prefill and batch decode over a paged KV cache (the form vLLM calls), in bf16 and
fp16, with grouped-query heads. The tolerances are fixed here, not tuned to the results: max |out - ref| at most
0.02 for bf16 and 0.005 for fp16. The workspace must hold compiled modules afterwards (the JIT really ran).
Exit status 0 only if every check passed.
"""

import logging
from os import environ
from pathlib import Path
from sys import exit as sys_exit

from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    single_decode_with_kv_cache,
    single_prefill_with_kv_cache,
)
from torch import Tensor, bfloat16, device, float16, float32, int32, manual_seed, ones, randn, tensor, tril, uint8, zeros
from torch import bool as torch_bool
from torch import dtype as torch_dtype
from torch.cuda import get_device_name, is_available

LOGGER = logging.getLogger("jit_kernel_check")
TOLERANCE = {bfloat16: 0.02, float16: 0.005}
QO_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128


class jit_kernel_check:
    """Runs JIT-compiled flashinfer attention kernels against a float32 reference."""

    @staticmethod
    def _reference(q: Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor:
        """Attention in float32. q [qo, Hq, D], k/v [kv, Hkv, D] -> [qo, Hq, D]; causal needs qo == kv."""
        group = q.shape[1] // k.shape[1]
        qf = q.to(float32).transpose(0, 1)
        kf = k.to(float32).repeat_interleave(group, dim=1).transpose(0, 1)
        vf = v.to(float32).repeat_interleave(group, dim=1).transpose(0, 1)
        scores = qf @ kf.transpose(1, 2) / (q.shape[-1] ** 0.5)
        if causal:
            keep = tril(ones(q.shape[0], k.shape[0], dtype=torch_bool, device=q.device))
            scores = scores.masked_fill(~keep, float("-inf"))
        return (scores.softmax(dim=-1) @ vf).transpose(0, 1)

    @staticmethod
    def _compare(name: str, out: Tensor, ref: Tensor, dtype: torch_dtype) -> bool:
        diff = (out.to(float32) - ref).abs().max().item()
        ok = tuple(out.shape) == tuple(ref.shape) and diff <= TOLERANCE[dtype]
        LOGGER.info(
            "  %-34s %-8s max_abs_diff=%.5f (tolerance %.3f) [%s]",
            name, str(dtype).removeprefix("torch."), diff, TOLERANCE[dtype], "OK" if ok else "FAIL",
        )
        return ok

    @staticmethod
    def check_single_decode(dtype: torch_dtype) -> bool:
        cuda = device("cuda")
        q = randn(QO_HEADS, HEAD_DIM, dtype=dtype, device=cuda)
        k = randn(512, KV_HEADS, HEAD_DIM, dtype=dtype, device=cuda)
        v = randn(512, KV_HEADS, HEAD_DIM, dtype=dtype, device=cuda)
        out = single_decode_with_kv_cache(q, k, v)
        ref = jit_kernel_check._reference(q.unsqueeze(0), k, v, causal=False).squeeze(0)
        return jit_kernel_check._compare("single_decode (kv 512)", out, ref, dtype)

    @staticmethod
    def check_single_prefill(dtype: torch_dtype) -> bool:
        cuda = device("cuda")
        q = randn(256, QO_HEADS, HEAD_DIM, dtype=dtype, device=cuda)
        k = randn(256, KV_HEADS, HEAD_DIM, dtype=dtype, device=cuda)
        v = randn(256, KV_HEADS, HEAD_DIM, dtype=dtype, device=cuda)
        out = single_prefill_with_kv_cache(q, k, v, causal=True)
        ref = jit_kernel_check._reference(q, k, v, causal=True)
        return jit_kernel_check._compare("single_prefill causal (qo=kv 256)", out, ref, dtype)

    @staticmethod
    def check_batch_decode(dtype: torch_dtype) -> bool:
        """Two requests over a paged NHD cache, page size 16: 3 pages (40 tokens) and 2 pages (21 tokens)."""
        cuda = device("cuda")
        page = 16
        wrapper = BatchDecodeWithPagedKVCacheWrapper(zeros(128 * 1024 * 1024, dtype=uint8, device=cuda), "NHD")
        indptr = tensor([0, 3, 5], dtype=int32, device=cuda)
        indices = tensor([4, 0, 2, 1, 3], dtype=int32, device=cuda)
        last_page_len = tensor([8, 5], dtype=int32, device=cuda)
        wrapper.plan(indptr, indices, last_page_len, QO_HEADS, KV_HEADS, HEAD_DIM, page, data_type=dtype, q_data_type=dtype)
        q = randn(2, QO_HEADS, HEAD_DIM, dtype=dtype, device=cuda)
        k_cache = randn(5, page, KV_HEADS, HEAD_DIM, dtype=dtype, device=cuda)
        v_cache = randn(5, page, KV_HEADS, HEAD_DIM, dtype=dtype, device=cuda)
        out = wrapper.run(q, (k_cache, v_cache))
        ok = True
        for request, (pages, last) in enumerate((([4, 0, 2], 8), ([1, 3], 5))):
            length = (len(pages) - 1) * page + last
            k = k_cache[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:length]
            v = v_cache[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:length]
            ref = jit_kernel_check._reference(q[request : request + 1], k, v, causal=False)
            ok = jit_kernel_check._compare(f"batch_decode paged (req {request}, kv {length})", out[request : request + 1], ref, dtype) and ok
        return ok

    @staticmethod
    def check_workspace_compiled() -> bool:
        base = Path(environ["FLASHINFER_WORKSPACE_BASE"])
        built = sorted(base.rglob("*.so"))
        LOGGER.info("shared objects JIT-compiled into the workspace: %d", len(built))
        for path in built:
            LOGGER.info("  %s", path.relative_to(base))
        return bool(built)

    @staticmethod
    def main() -> int:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        if environ.get("FLASHINFER_DISABLE_JIT") == "1":
            LOGGER.error("FLASHINFER_DISABLE_JIT=1 is set; this check needs the JIT")
            return 1
        if "FLASHINFER_WORKSPACE_BASE" not in environ:
            LOGGER.error("FLASHINFER_WORKSPACE_BASE must name an empty directory")
            return 1
        if not is_available():
            LOGGER.error("no CUDA device")
            return 1
        LOGGER.info("device: %s", get_device_name(0))
        manual_seed(0)
        failed = 0
        checks = [
            (check, dtype)
            for check in (
                jit_kernel_check.check_single_decode,
                jit_kernel_check.check_single_prefill,
                jit_kernel_check.check_batch_decode,
            )
            for dtype in (bfloat16, float16)
        ]
        for check, dtype in checks:
            try:
                ok = check(dtype)
            except Exception:
                LOGGER.exception("%s(%s) raised", check.__name__, dtype)
                ok = False
            failed += 0 if ok else 1
        try:
            ok = jit_kernel_check.check_workspace_compiled()
        except Exception:
            LOGGER.exception("check_workspace_compiled raised")
            ok = False
        failed += 0 if ok else 1
        LOGGER.info("jit_kernel_check: %s", "passed" if failed == 0 else f"FAILED ({failed})")
        return 1 if failed else 0


if __name__ == "__main__":
    sys_exit(jit_kernel_check.main())
