"""Run small models through vLLM and report which flashinfer modules were not served from the jit-cache.

Each model runs in its own process with an empty FLASHINFER_WORKSPACE_BASE and JIT allowed, so every flashinfer module
the model needed that is not in the installed jit-cache is compiled into that workspace. The compiled names are the
gaps. This is a measurement for the survey (tuned/survey/hf_shape_survey.py gives the config-implied shapes); it is
not a release gate. Entries are MODEL or MODEL@ATTENTION_BACKEND (for example Qwen/Qwen3-0.6B@FLASHINFER).

Run with the vllm build venv's python, the packaged wheels installed in it:

    python tuned/survey/run_small_models.py --out gaps.jsonl Qwen/Qwen3-0.6B Qwen/Qwen3-0.6B@FLASHINFER
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger("run_small_models")

CHILD = """
import sys
from vllm import LLM, SamplingParams
model, backend, util = sys.argv[1], sys.argv[2], float(sys.argv[3])
kwargs = {"attention_backend": backend} if backend else {}
llm = LLM(model=model, gpu_memory_utilization=util, max_model_len=512, enforce_eager=True, **kwargs)
out = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=16, temperature=0.0))
print("RESULT_OK:", repr(out[0].outputs[0].text))
"""


class run_small_models:
    """Runs each entry in a child process and lists the flashinfer modules compiled at run time."""

    @staticmethod
    def compiled_modules(workspace: Path) -> list[str]:
        names = {p.parent.name for p in workspace.rglob("*.so") if "cached_ops" in p.parts}
        return sorted(names)

    @staticmethod
    def run_one(entry: str, util: float, timeout: int) -> dict[str, Any]:
        model, _, backend = entry.partition("@")
        with tempfile.TemporaryDirectory() as workspace:
            env = {**os.environ, "FLASHINFER_WORKSPACE_BASE": workspace}
            env.pop("FLASHINFER_DISABLE_JIT", None)
            # The flash-attn build is native sm_121a with no PTX; without this vLLM refuses the FLASH_ATTN backend on a
            # driver older than the toolkit (same setting as vllm's tuned/run_gb10.sh and gb10_acceptance_test.sh).
            env.setdefault("VLLM_FLASH_ATTN_NO_PTX", "1")
            try:
                done = subprocess.run(
                    [sys.executable, "-I", "-c", CHILD, model, backend, str(util)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                output = done.stdout + done.stderr
                ok = done.returncode == 0 and "RESULT_OK:" in done.stdout
                errors = [ln for ln in output.splitlines() if "Error" in ln and "HTTP" not in ln]
                error: Optional[str] = None if ok else (errors[0] if errors else output.strip().splitlines()[-1])[:300]
            except subprocess.TimeoutExpired:
                ok, error = False, f"timeout after {timeout}s"
            return {
                "entry": entry,
                "ok": ok,
                "error": error,
                "compiled": run_small_models.compiled_modules(Path(workspace)),
            }

    @staticmethod
    def main() -> int:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        parser = ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("entries", nargs="+")
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--util", type=float, default=0.15)
        parser.add_argument("--timeout", type=int, default=1500)
        args = parser.parse_args()
        rows = []
        for entry in args.entries:
            row = run_small_models.run_one(entry, args.util, args.timeout)
            LOGGER.info("%s ok=%s compiled=%d %s", entry, row["ok"], len(row["compiled"]), row["error"] or "")
            rows.append(row)
            args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(run_small_models.main())
