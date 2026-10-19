"""Run the ops a tuned flashinfer build promises to have precompiled, with JIT disabled and an empty workspace.

Run from an empty directory against an installed pair of wheels (flashinfer-python and flashinfer-jit-cache):

    FLASHINFER_DISABLE_JIT=1 FLASHINFER_WORKSPACE_BASE=<empty dir> python -I jit_cache_coverage.py

The XQA shapes come from the same FLASHINFER_AOT_XQA_* lists the jit-cache was built from, so this checks the
build's own configuration. Any op missing from the cache raises MissingJITCacheError; nothing may be compiled
into the workspace. Exit status 0 only if every check passed.
"""

import logging
from importlib.metadata import version
from itertools import product
from os import environ
from pathlib import Path
from sys import exit as sys_exit

from flashinfer import __file__ as flashinfer_file
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.jit.tllm_utils import gen_trtllm_utils_module
from flashinfer.xqa import get_xqa_module
from torch import bfloat16, device, float8_e4m3fn, float16, int32, randn, tensor, uint8, zeros
from torch import dtype as torch_dtype
from torch.cuda import is_available

LOGGER = logging.getLogger("jit_cache_coverage")
INPUT_TYPES = {"bfloat16": bfloat16, "float16": float16}


class jit_cache_coverage:
    """Checks that the installed jit-cache covers the configured ops without compiling."""

    @staticmethod
    def _csv(name: str, default: str) -> list[str]:
        return [item for item in environ.get(name, default).split(",") if item]

    @staticmethod
    def check_pairing() -> bool:
        python = version("flashinfer-python")
        cache = version("flashinfer-jit-cache")
        LOGGER.info("flashinfer-python %s from %s; flashinfer-jit-cache %s", python, flashinfer_file, cache)
        return python == cache

    @staticmethod
    def check_trtllm_utils() -> bool:
        gen_trtllm_utils_module().build_and_load()
        LOGGER.info("trtllm_utils loaded from the cache")
        return True

    @staticmethod
    def check_xqa() -> bool:
        types = jit_cache_coverage._csv("FLASHINFER_AOT_XQA_INPUT_TYPE", "")
        fp8 = jit_cache_coverage._csv("FLASHINFER_AOT_XQA_FP8_KV_CACHE", "")
        pages = jit_cache_coverage._csv("FLASHINFER_AOT_XQA_TOKEN_PER_PAGE", "")
        heads = jit_cache_coverage._csv("FLASHINFER_AOT_XQA_HEAD_SIZE", "")
        groups = jit_cache_coverage._csv("FLASHINFER_AOT_XQA_HEAD_GRP_SIZE", "")
        if not (types and fp8 and pages and heads and groups):
            LOGGER.error("the FLASHINFER_AOT_XQA_* lists are not set; nothing to check")
            return False
        count = 0
        for kind, quantised, page, head, group in product(types, fp8, pages, heads, groups):
            input_dtype: torch_dtype = INPUT_TYPES[kind]
            kv_dtype: torch_dtype = float8_e4m3fn if quantised == "true" else input_dtype
            get_xqa_module(input_dtype, kv_dtype, int(page), int(head), int(group), False, input_dtype, 1)
            count += 1
        LOGGER.info("%d xqa modules loaded from the cache", count)
        return True

    @staticmethod
    def check_decode() -> bool:
        """Batch decode over a paged KV cache: the form vLLM calls. (single_decode is not in the AOT set.)"""
        if not is_available():
            LOGGER.error("no CUDA device")
            return False
        cuda = device("cuda")
        wrapper = BatchDecodeWithPagedKVCacheWrapper(zeros(128 * 1024 * 1024, dtype=uint8, device=cuda), "NHD")
        indptr = tensor([0, 1], dtype=int32, device=cuda)
        indices = tensor([0], dtype=int32, device=cuda)
        last_page_len = tensor([4], dtype=int32, device=cuda)
        wrapper.plan(indptr, indices, last_page_len, 8, 8, 128, 16, data_type=bfloat16, q_data_type=bfloat16)
        q = randn(1, 8, 128, dtype=bfloat16, device=cuda)
        kv = (randn(1, 16, 8, 128, dtype=bfloat16, device=cuda), randn(1, 16, 8, 128, dtype=bfloat16, device=cuda))
        out = wrapper.run(q, kv)
        LOGGER.info("BatchDecodeWithPagedKVCacheWrapper ran: %s %s", tuple(out.shape), out.dtype)
        return tuple(out.shape) == (1, 8, 128)

    @staticmethod
    def check_workspace_empty() -> bool:
        base = Path(environ["FLASHINFER_WORKSPACE_BASE"])
        built = sorted(str(p) for p in base.rglob("*.so"))
        LOGGER.info("shared objects compiled into the workspace: %d", len(built))
        for path in built:
            LOGGER.error("compiled at run time: %s", path)
        return not built

    @staticmethod
    def main() -> int:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        if environ.get("FLASHINFER_DISABLE_JIT") != "1":
            LOGGER.error("FLASHINFER_DISABLE_JIT=1 must be set; otherwise a cache miss compiles silently")
            return 1
        failed = 0
        for check in (
            jit_cache_coverage.check_pairing,
            jit_cache_coverage.check_trtllm_utils,
            jit_cache_coverage.check_xqa,
            jit_cache_coverage.check_decode,
            jit_cache_coverage.check_workspace_empty,
        ):
            try:
                ok = check()
            except Exception:
                LOGGER.exception("%s raised", check.__name__)
                ok = False
            LOGGER.info("%s: %s", check.__name__, "ok" if ok else "FAIL")
            failed += 0 if ok else 1
        LOGGER.info("jit_cache_coverage: %s", "passed" if failed == 0 else f"FAILED ({failed})")
        return 1 if failed else 0


if __name__ == "__main__":
    sys_exit(jit_cache_coverage.main())
