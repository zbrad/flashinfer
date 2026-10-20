# Which model shapes does the GB10 flashinfer jit-cache need to cover?

Survey of 2026-10-19. The release `0.7.0+gb10.cu134.20261019.0` ships a jit-cache scoped to two target models. This note
asks what else people run, what that would need from flashinfer, and whether that justifies more work. It is a first pass:
the sample is small and the limits are listed at the end.

How to read the provenance labels: **config** = read from a model's `config.json` on Hugging Face by
`tuned/survey/hf_shape_survey.py`; **measured** = observed by running the model (`tuned/survey/run_small_models.py`, vLLM
`0.32.1.dev88`, packaged wheels from this release); **inferred** = Claude's reasoning, not checked; **unmeasured** = not run.

## What the cache ships today (config of the build: `tuned/repo.env`)

- Attention (FA2/FA3 prefill and batch decode): bf16, `head_dim` 128 (FA3 also 192/128), no fp16.
- XQA decode: bf16 input, bf16 or fp8 e4m3 KV cache, pages 16/32/64, `head_dim` 128, head-group ratio 8 and 16, no
  speculative decode, no sliding window.
- Mamba state update, norm/rope/sampling, MoE, activation, `trtllm_utils`.

Anything else is compiled at first use, or fails under `FLASHINFER_DISABLE_JIT=1`.

## 1. What the models look like (config)

The 300 most downloaded `text-generation` models at the time; 225 had a `config.json` and an architecture in vLLM's registry
(5 were not in the registry; the other 70 had no `config.json`, mostly GGUF repositories). Counts are of those 225
(`tuned/survey/data/hf_survey_2026-10-19.jsonl`). A download ranking includes fine-tunes and quantized copies of the same
base, so these are counts of popular repositories, not of distinct architectures.

| Field | Values (number of models) |
|---|---|
| dtype in config | bfloat16 183, float16 26, float32 5, not stated 11 |
| quantization | none 154, fp8 19, awq 17, modelopt 16, compressed-tensors 11, mxfp4 3, gptq 3, other 2 |
| `head_dim` | 128: 125, 256: 37, 64: 31, 56/192/512 (MLA-style): 12, 96/112/102/80/32/4/2: 14, unknown 6 |
| head-group ratio | 1: 43, 4: 42, 8: 37, 6: 21, 2: 17, 7: 17, 5: 16, 16: 14, 3: 8, 64/12/32/128: 6, unknown 4 |
| features | sliding window 65, MLA 19, Mamba or hybrid 38 (`head_dim` 128 or 256), MoE 69 |

- **Config-implied match with the shipped XQA shape** (`head_dim` 128 and ratio 8 or 16): 33 of 225. The other shapes
  that matter in count are ratio 4 (42) and ratio 1 (43), and `head_dim` 64 (31) and 256 (37). *Whether those models would ever
  call flashinfer is a separate question; see section 2.*
- float16 is about one model in nine in this sample: 26 models, 19 of them AWQ or GPTQ copies, 16 with Qwen in the name (config).

## 2. What running them shows (measured)

The runner starts each model with an empty flashinfer workspace and JIT allowed, so the modules it had to compile are
exactly the modules the installed jit-cache lacks. Single GPU, `gpu_memory_utilization` 0.15, `max_model_len` 512, eager
mode, 16 generated tokens, `VLLM_FLASH_ATTN_NO_PTX=1`.

**vLLM's default attention backend on GB10 (flash-attn)**: every model loaded and generated, with **zero flashinfer
modules compiled**.

| Model | dtype / quant | `head_dim` | group ratio | modules compiled |
|---|---|---|---|---|
| Qwen/Qwen3-0.6B | bf16 | 128 | 2 | 0 |
| Qwen/Qwen2.5-0.5B-Instruct | bf16 | 64 | 7 | 0 |
| HuggingFaceTB/SmolLM2-135M | bf16 | 64 | 3 | 0 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | bf16 | 64 | 8 | 0 |
| meta-llama/Llama-3.2-1B-Instruct | bf16 | 64 | 4 | 0 |
| google/gemma-3-1b-it | bf16, sliding window | 256 | 4 | 0 |
| Qwen/Qwen3-4B-Instruct-2507-FP8 | fp8 weights | 128 | 4 | 0 |
| facebook/opt-125m | fp16 | 64 | 1 | 0 |
| Qwen/Qwen2.5-0.5B-Instruct-AWQ | awq | 64 | 7 | 0 |

(Group ratios are the config values from section 1's data file; the AWQ row and the opt row were run in the second batch,
`gaps_flashinfer_backend_2026-10-19.jsonl`.)

**vLLM's FlashInfer attention backend (`attention_backend="FLASHINFER"`)**: every model loaded and generated, and each
compiled the modules below. This is the case that matters if a user selects flashinfer for attention.

| Model | `head_dim` | ratio | modules compiled (not in the cache) |
|---|---|---|---|
| Qwen/Qwen3-0.6B | 128 | 2 | xqa (ratio 2) |
| Qwen/Qwen3-4B-Instruct-2507-FP8 | 128 | 4 | xqa (ratio 4) |
| Qwen/Qwen2.5-0.5B-Instruct | 64 | 7 | prefill (`head_dim` 64), xqa (64, ratio 7) |
| HuggingFaceTB/SmolLM2-135M | 64 | 3 | prefill (64), xqa (64, ratio 3) |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 64 | 8 | prefill (64), xqa (64, ratio 8) |
| meta-llama/Llama-3.2-1B-Instruct | 64 | 4 | prefill (64), xqa (64, ratio 4) |
| google/gemma-3-1b-it | 256 | 4 | prefill (256) with and without sliding window, xqa (256, ratio 4) with and without sliding window |

Prefill at `head_dim` 128 was served from the cache in the two 128-dim runs; XQA at 128 was served only for ratio 8 and 16.

## 3. What this suggests

- **Default backend: no work.** Nothing in the 9 default-backend runs touched the cache. *Inferred:* dense bf16/fp8/fp16/AWQ
  models on the default backend need no new flashinfer modules, whatever their shapes. The measurement covers only the nine models above.
- **FlashInfer attention backend: the gap is per (`head_dim`, group ratio)** for XQA decode and per `head_dim` for prefill. In
  this sample, `head_dim` 64 and 256 (all ratios seen: 3, 4, 7, 8 and 4) and `head_dim` 128 with ratios 2 and 4 all compile at first use. Under `FLASHINFER_DISABLE_JIT=1` they
  would fail. *Inferred, not run:* with JIT disabled these models would raise `MissingJITCacheError`; the runs above had JIT allowed.
- **A cheap targeted add would be**: XQA for `head_dim` {64, 128, 256} × ratios {1, 2, 3, 4, 5, 6, 7, 8, 16} (the ratios seen in
  section 1) at the existing pages and KV dtypes, plus prefill at `head_dim` 64 and 256. *Not measured:* the build time and
  size this adds. The first number to get is the module count and the wheel size delta from one trial build, before deciding.
- **Not covered by this survey** and each deserving its own measurement:
  1. the Qwen3.5/3.6 and similar hybrids (`head_dim` 256 with linear attention; 38 Mamba/hybrid configs, with ratios 16 (12), 4 (8), 6 (8), 8 (7), 5 (2), 32 (1));
  2. MLA models (DeepSeek, GLM: 19 configs);
  3. MoE and NVFP4 models other than Nemotron (modelopt 16, mxfp4 3);
  4. fp8 KV cache with the FlashInfer backend on non-128 head dims;
  5. speculative decoding and sliding window at XQA (the 4 modules compiled by the flashinfer test slice in the release notes);
  6. long contexts and large batches (`max_model_len` was 512).

## 4. Small models worth keeping as stand-ins

The nine above already load on GB10 and are small (135M to 4B). *Inferred:* a stand-in set for regression should cover each
distinct (`head_dim`, ratio, dtype) the cache claims, using these:

| Shape | Stand-in (already run) |
|---|---|
| 64 / ratio 3, 4, 7, 8 | SmolLM2-135M, Llama-3.2-1B, Qwen2.5-0.5B, TinyLlama |
| 128 / ratio 2, 4 | Qwen3-0.6B, Qwen3-4B-FP8 |
| 256 / ratio 4, sliding window | gemma-3-1b-it |
| fp16, AWQ | opt-125m, Qwen2.5-0.5B-AWQ |

Llama-3.2-1B and gemma-3-1b are gated on Hugging Face; they worked with the token on this machine, but not for anyone
without access. Ungated alternatives exist for most rows.

## 5. Limits of this survey

- The sample is the most downloaded 300, one day, and mixes bases with their copies. The ranking says nothing about the users of this fleet.
- Nine models were run (seven on both backends, opt-125m and the AWQ model on the default backend only), 16 tokens each, 512 context, eager mode. A larger batch or a
  CUDA-graph run could reach other modules.
- The runs allowed JIT, so a gap shows as a compiled module. A run with JIT disabled would show the first missing
  module only.
- Shape counts are from `config.json` and the derivation in `hf_shape_survey.py` (group ratio = heads / KV heads, head_dim =
  `head_dim` or hidden size / heads). Models with unusual configs (nested text configs, multi-query flags) may be read wrongly.
- `tuned/survey/data/hf_survey_2026-10-19.jsonl` is a snapshot; rerunning the script later gives a different list.

## 6. Reproduce

```
# 1. config survey (any python with huggingface_hub and vllm; HF_TOKEN from the environment if set)
python tuned/survey/hf_shape_survey.py --limit 300 --out survey.jsonl
# 2. measured gaps (vllm build venv python, this release's wheels installed in it; MODEL@BACKEND selects the attention backend)
python tuned/survey/run_small_models.py --out gaps.jsonl Qwen/Qwen3-0.6B Qwen/Qwen3-0.6B@FLASHINFER ...
```
