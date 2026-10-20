"""Survey Hugging Face text-generation models for the attention shapes a flashinfer jit-cache would have to cover.

For the most downloaded text-generation models whose architecture vLLM supports, read config.json and derive the
shape facts that decide which flashinfer modules a deployment can hit: head_dim, head-group ratio, dtype,
quantization, sliding window, MLA, Mamba, MoE. One JSON line per model goes to the output file, and a summary is
logged.

This is config-implied, not measured. Whether flashinfer is used at all depends on the backend vLLM picks on the
target GPU, so a row says what shape the model would present, not that the cache must serve it. The measured
check is tuned/tests/extra_tests.sh and a run of the model with FLASHINFER_DISABLE_JIT=1.

Run (any python with huggingface_hub and vllm, for example the vllm build venv; HF_TOKEN is read from the environment
by huggingface_hub if set):

    python tuned/survey/hf_shape_survey.py --limit 300 --out survey.jsonl
"""

import json
import logging
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import HfApi, hf_hub_download

LOGGER = logging.getLogger("hf_shape_survey")


class hf_shape_survey:
    """Collects and summarizes config-implied attention shapes."""

    # What tuned/repo.env ships for gb10 today (XQA): head_dim and head-group ratios.
    SHIPPED_HEAD_DIMS = frozenset({128})
    SHIPPED_GROUP_RATIOS = frozenset({8, 16})
    NESTED_TEXT_KEYS = ("text_config", "llm_config", "language_config")

    @staticmethod
    def supported_architectures() -> frozenset[str]:
        from vllm.model_executor.models import ModelRegistry

        return frozenset(ModelRegistry.get_supported_archs())

    @staticmethod
    def top_models(limit: int) -> list[tuple[str, int]]:
        api = HfApi()
        found = api.list_models(
            pipeline_tag="text-generation",
            sort="downloads",
            limit=limit,
            expand=["downloads"],
        )
        return [(m.id, int(m.downloads or 0)) for m in found]

    @staticmethod
    def fetch_config(model_id: str) -> Optional[dict[str, Any]]:
        try:
            path = hf_hub_download(model_id, "config.json")
            return json.loads(Path(path).read_text())
        except Exception as error:  # gated, missing or not JSON: recorded, not fatal
            LOGGER.warning("%s: no config.json (%s)", model_id, type(error).__name__)
            return None

    @staticmethod
    def derive(model_id: str, downloads: int, config: dict[str, Any]) -> dict[str, Any]:
        text = config
        for key in hf_shape_survey.NESTED_TEXT_KEYS:
            if isinstance(config.get(key), dict):
                text = {**config, **config[key]}
                break
        heads = text.get("num_attention_heads")
        kv_heads = text.get("num_key_value_heads", heads)
        hidden = text.get("hidden_size")
        head_dim = text.get("head_dim")
        if head_dim is None and heads and hidden:
            head_dim = hidden // heads
        ratio = heads // kv_heads if heads and kv_heads and heads % kv_heads == 0 else None
        quant = config.get("quantization_config") or {}
        layer_types = text.get("layer_types") or []
        experts = (
            text.get("num_experts")
            or text.get("num_local_experts")
            or text.get("n_routed_experts")
        )
        arch = (config.get("architectures") or [None])[0]
        return {
            "model": model_id,
            "downloads": downloads,
            "architecture": arch,
            "model_type": config.get("model_type"),
            "dtype": text.get("torch_dtype") or text.get("dtype"),
            "quant_method": quant.get("quant_method"),
            "heads": heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "group_ratio": ratio,
            "sliding_window": bool(text.get("sliding_window"))
            or "sliding_attention" in layer_types,
            "mla": "kv_lora_rank" in text,
            "mamba": any(k.startswith("mamba") or k == "ssm_state_size" for k in text),
            "mamba_head_dim": text.get("mamba_head_dim"),
            "moe_experts": experts,
            "xqa_shape_shipped": head_dim in hf_shape_survey.SHIPPED_HEAD_DIMS
            and ratio in hf_shape_survey.SHIPPED_GROUP_RATIOS,
        }

    @staticmethod
    def summarize(rows: list[dict[str, Any]]) -> None:
        LOGGER.info("models with a supported architecture and a config: %d", len(rows))
        for field in ("dtype", "quant_method", "head_dim", "group_ratio", "mamba_head_dim"):
            LOGGER.info("%s: %s", field, dict(Counter(str(r[field]) for r in rows).most_common()))
        for field in ("sliding_window", "mla", "mamba", "xqa_shape_shipped"):
            LOGGER.info("%s: %d", field, sum(1 for r in rows if r[field]))
        LOGGER.info("moe: %d", sum(1 for r in rows if r["moe_experts"]))

    @staticmethod
    def main() -> int:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        parser = ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("--limit", type=int, default=300)
        parser.add_argument("--out", type=Path, required=True)
        args = parser.parse_args()
        supported = hf_shape_survey.supported_architectures()
        rows: list[dict[str, Any]] = []
        skipped_arch = 0
        for model_id, downloads in hf_shape_survey.top_models(args.limit):
            config = hf_shape_survey.fetch_config(model_id)
            if config is None:
                continue
            row = hf_shape_survey.derive(model_id, downloads, config)
            if row["architecture"] not in supported:
                skipped_arch += 1
                continue
            rows.append(row)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        LOGGER.info("skipped (architecture not in vLLM's registry): %d", skipped_arch)
        hf_shape_survey.summarize(rows)
        return 0


if __name__ == "__main__":
    raise SystemExit(hf_shape_survey.main())
