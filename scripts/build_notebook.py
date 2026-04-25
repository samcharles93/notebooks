#!/usr/bin/env python3
# Builds /work/clones/notebooks/mantle_sft.ipynb by constructing the JSON
# programmatically. Forks the upstream Unsloth Qwen3.5-4B Vision SFT notebook
# topology but rewrites it for text-only Qwen3.5-4B-Instruct on the
# samcharles93/mantle-sft private dataset, with portable Colab/Runpod/HF-Jobs
# bootstrapping, immediate-push checkpoints, and the locked hyperparameters
# from configs/qwen3.5-4b-sft.yaml.

import json
import sys
from pathlib import Path

OUT = Path("/work/clones/notebooks/mantle_sft.ipynb")


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l if l.endswith("\n") else l + "\n" for l in lines],
    }


def code(*lines: str) -> dict:
    src = list(lines)
    src = [l if l.endswith("\n") else l + "\n" for l in src]
    if src:
        src[-1] = src[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


cells: list[dict] = []

cells.append(md(
    "# Mantle SFT — Qwen3.5-4B-Instruct LoRA",
    "",
    "Single-stage SFT of `Qwen/Qwen3.5-4B-Instruct` on the private `samcharles93/mantle-sft`",
    "corpus (182 sessions, ~77% with `<think>` reasoning). Forked from the Unsloth",
    "Qwen3.5-4B Vision SFT notebook and rewritten for text-only training.",
    "",
    "**Targets**: Colab T4 (free) · Colab Pro+ A100 · Runpod RTX 6000 Ada / A100-80G · HF Jobs.",
    "**Output**: LoRA adapter at `samcharles93/qwen3.5-4b-mantle-sft` and merged bf16 at",
    "`samcharles93/qwen3.5-4b-mantle-sft-merged` (both private). Pushes happen as soon as each",
    "artefact is ready so a Colab disconnect does not lose work.",
    "",
    "Hyperparameters are locked from `scratchpad/mantle-finetune/configs/qwen3.5-4b-sft.yaml`:",
    "r=32, α=64, bf16 LoRA, seq 8192, 3 epochs, LR 2e-4, packing=false, `enable_thinking=True`,",
    "`train_on_responses_only=True`. QLoRA is NOT used (Unsloth does not recommend it for Qwen3.5).",
))

cells.append(md(
    "## 1. Environment",
    "",
    "Auto-detects Colab / Runpod / Jobs / local and chooses a writable `WORK` directory. Fails",
    "loud on a too-small GPU rather than half-running and OOMing 30 minutes in.",
))

cells.append(code(
    "import os, sys, shutil, subprocess, pathlib",
    "",
    "def detect_work() -> pathlib.Path:",
    "    if pathlib.Path('/workspace').exists() and os.access('/workspace', os.W_OK):",
    "        return pathlib.Path('/workspace/mantle-sft')",
    "    if pathlib.Path('/content').exists() and os.access('/content', os.W_OK):",
    "        return pathlib.Path('/content/mantle-sft')",
    "    return pathlib.Path.home() / 'mantle-sft'",
    "",
    "WORK = detect_work()",
    "WORK.mkdir(parents=True, exist_ok=True)",
    "os.chdir(WORK)",
    "print(f'WORK = {WORK}')",
    "print(f'cwd  = {os.getcwd()}')",
    "",
    "try:",
    "    out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip()",
    "    print(f'GPU  = {out}')",
    "    mem_mib = int(out.split(',')[1].strip().split()[0])",
    "    if mem_mib < 15000:",
    "        raise RuntimeError(f'GPU has only {mem_mib} MiB; need >=16 GiB for Qwen3.5-4B bf16 LoRA. Aborting.')",
    "except FileNotFoundError:",
    "    raise RuntimeError('nvidia-smi not found; this notebook requires a CUDA GPU.')",
))

cells.append(md(
    "## 2. Install dependencies",
    "",
    "Pip-based (not uv) so every host behaves identically. Pinned to the floor versions verified",
    "in `configs/qwen3.5-4b-sft.yaml`: `transformers>=5.0.0`, `trl>=0.28.0`, `unsloth>=2026.2.26`",
    "(commit `f9d4a53` — day-zero Qwen3.5 support).",
))

cells.append(code(
    "%pip install -q --upgrade pip",
    "%pip install -q 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'",
    "%pip install -q 'transformers>=5.0.0' 'trl>=0.28.0' 'accelerate>=1.2.0' 'peft>=0.14.0' 'bitsandbytes>=0.45.0' 'datasets>=3.2.0' 'huggingface_hub>=0.27.0'",
))

cells.append(md(
    "## 3. Hugging Face authentication",
    "",
    "Reads the token from (in order) `HF_TOKEN` env var, Colab secrets, or an interactive prompt.",
    "The token must have **read** access to `samcharles93/mantle-sft` (private dataset) and **write**",
    "access to `samcharles93/qwen3.5-4b-mantle-sft*` (private models).",
))

cells.append(code(
    "import os",
    "from huggingface_hub import login, whoami",
    "",
    "def resolve_token() -> str:",
    "    tok = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')",
    "    if tok:",
    "        return tok",
    "    try:",
    "        from google.colab import userdata",
    "        tok = userdata.get('HF_TOKEN')",
    "        if tok:",
    "            return tok",
    "    except Exception:",
    "        pass",
    "    from getpass import getpass",
    "    return getpass('HF token (write scope): ').strip()",
    "",
    "HF_TOKEN = resolve_token()",
    "assert HF_TOKEN, 'HF_TOKEN is required'",
    "login(token=HF_TOKEN, add_to_git_credential=False)",
    "os.environ['HF_TOKEN'] = HF_TOKEN",
    "print('Logged in as:', whoami()['name'])",
))

cells.append(md(
    "## 4. Constants",
    "",
    "All knobs that diverge from the upstream Unsloth notebook are pinned here.",
))

cells.append(code(
    "BASE_MODEL          = 'Qwen/Qwen3.5-4B-Instruct'",
    "DATASET_REPO        = 'samcharles93/mantle-sft'",
    "DATASET_FILE        = 'sft.jsonl'",
    "ADAPTER_REPO        = 'samcharles93/qwen3.5-4b-mantle-sft'",
    "MERGED_REPO         = 'samcharles93/qwen3.5-4b-mantle-sft-merged'",
    "",
    "MAX_SEQ_LENGTH      = 8192",
    "LORA_R              = 32",
    "LORA_ALPHA          = 64",
    "LORA_DROPOUT        = 0.0",
    "TARGET_MODULES      = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']",
    "",
    "NUM_EPOCHS          = 3",
    "PER_DEVICE_BATCH    = 2",
    "GRAD_ACCUM_STEPS    = 4",
    "LEARNING_RATE       = 2e-4",
    "WARMUP_STEPS        = 5",
    "WEIGHT_DECAY        = 0.01",
    "MAX_GRAD_NORM       = 1.0",
    "SEED                = 3407",
    "",
    "ENABLE_THINKING     = True",
    "PACKING             = False",
    "",
    "OUTPUT_DIR          = str(WORK / 'runs' / 'qwen3.5-4b-sft')",
    "MERGED_DIR          = str(WORK / 'runs' / 'qwen3.5-4b-merged')",
    "for d in (OUTPUT_DIR, MERGED_DIR):",
    "    pathlib.Path(d).mkdir(parents=True, exist_ok=True)",
))

cells.append(md(
    "## 5. Load model and tokenizer",
    "",
    "`FastLanguageModel.from_pretrained` returns the bf16 base + a tokenizer with the Qwen3.5",
    "chat template already wired. `dtype=None` lets Unsloth pick (bf16 on Ampere+).",
))

cells.append(code(
    "from unsloth import FastLanguageModel",
    "import torch",
    "",
    "model, tokenizer = FastLanguageModel.from_pretrained(",
    "    model_name      = BASE_MODEL,",
    "    max_seq_length  = MAX_SEQ_LENGTH,",
    "    dtype           = None,",
    "    load_in_4bit    = False,",
    "    load_in_8bit    = False,",
    "    full_finetuning = False,",
    "    token           = HF_TOKEN,",
    ")",
    "print('Loaded:', BASE_MODEL)",
    "print('Pad token:', tokenizer.pad_token, ' EOS:', tokenizer.eos_token)",
))

cells.append(md(
    "## 6. Attach LoRA adapter",
    "",
    "r=32, α=64 (scale 2.0). The corpus is small (182 sessions) and narrow (Mantle/Go/MCF), so",
    "we override Unsloth's r=16 default to give the adapter more capacity. `dropout=0` because",
    "small datasets do not benefit from dropout. Gradient checkpointing uses Unsloth's custom",
    "kernel — about 30% less VRAM than HF's default.",
))

cells.append(code(
    "model = FastLanguageModel.get_peft_model(",
    "    model,",
    "    r                          = LORA_R,",
    "    lora_alpha                 = LORA_ALPHA,",
    "    lora_dropout               = LORA_DROPOUT,",
    "    target_modules             = TARGET_MODULES,",
    "    bias                       = 'none',",
    "    use_gradient_checkpointing = 'unsloth',",
    "    random_state               = SEED,",
    "    use_rslora                 = False,",
    "    loftq_config               = None,",
    ")",
    "model.print_trainable_parameters()",
))

cells.append(md(
    "## 7. Verify dataset exists, then load",
    "",
    "Fails fast if the private dataset has not been uploaded yet (avoids a 30-second silent wait",
    "on a 401). `data_files=DATASET_FILE` pins us to the exact JSONL we built — `mantle-sft` may",
    "later contain other splits.",
))

cells.append(code(
    "from huggingface_hub import HfApi",
    "from datasets import load_dataset",
    "",
    "api = HfApi(token=HF_TOKEN)",
    "files = api.list_repo_files(repo_id=DATASET_REPO, repo_type='dataset')",
    "assert DATASET_FILE in files, f'{DATASET_FILE} not found in {DATASET_REPO}; files={files}'",
    "print(f'Dataset OK: {DATASET_REPO}/{DATASET_FILE}')",
    "",
    "raw = load_dataset(DATASET_REPO, data_files=DATASET_FILE, split='train', token=HF_TOKEN)",
    "print(f'Loaded {len(raw)} sessions')",
    "print('Columns:', raw.column_names)",
    "print('Sample roles:', [m['role'] for m in raw[0]['messages'][:4]])",
))

cells.append(md(
    "## 8. Apply Qwen3.5 chat template",
    "",
    "Each row already has a `messages` array of `{role, content}` (with optional",
    "`reasoning_content` for thinking turns). `apply_chat_template` produces the model-ready",
    "string per row; `enable_thinking=True` preserves `<think>...</think>` blocks for the 128",
    "thinking sessions and emits an empty `<think>\\n\\n</think>` for the 54 plain ones.",
))

cells.append(code(
    "def format_row(row):",
    "    text = tokenizer.apply_chat_template(",
    "        row['messages'],",
    "        tokenize         = False,",
    "        add_generation_prompt = False,",
    "        enable_thinking  = ENABLE_THINKING,",
    "    )",
    "    return {'text': text}",
    "",
    "formatted = raw.map(format_row, remove_columns=raw.column_names, num_proc=2)",
    "print('Formatted rows:', len(formatted))",
    "print('--- sample (first 600 chars) ---')",
    "print(formatted[0]['text'][:600])",
))

cells.append(md(
    "## 9. SFTConfig + trainer",
    "",
    "Vision-specific args from the upstream notebook (`dataset_text_field=None`, image collator,",
    "`remove_unused_columns=False`) are dropped. We use the standard text path: `dataset_text_field='text'`,",
    "no custom collator. `train_on_responses_only` will mask the prompt tokens after the trainer",
    "is built.",
))

cells.append(code(
    "from trl import SFTConfig, SFTTrainer",
    "",
    "cfg = SFTConfig(",
    "    output_dir                  = OUTPUT_DIR,",
    "    num_train_epochs            = NUM_EPOCHS,",
    "    per_device_train_batch_size = PER_DEVICE_BATCH,",
    "    gradient_accumulation_steps = GRAD_ACCUM_STEPS,",
    "    learning_rate               = LEARNING_RATE,",
    "    lr_scheduler_type           = 'linear',",
    "    warmup_steps                = WARMUP_STEPS,",
    "    weight_decay                = WEIGHT_DECAY,",
    "    max_grad_norm               = MAX_GRAD_NORM,",
    "    optim                       = 'adamw_8bit',",
    "    bf16                        = True,",
    "    fp16                        = False,",
    "    seed                        = SEED,",
    "    logging_steps               = 10,",
    "    save_steps                  = 100,",
    "    save_total_limit            = 2,",
    "    report_to                   = 'none',",
    "    dataset_text_field          = 'text',",
    "    max_length                  = MAX_SEQ_LENGTH,",
    "    packing                     = PACKING,",
    "    dataset_num_proc            = 2,",
    ")",
    "",
    "trainer = SFTTrainer(",
    "    model           = model,",
    "    tokenizer       = tokenizer,",
    "    train_dataset   = formatted,",
    "    args            = cfg,",
    ")",
))

cells.append(md(
    "## 10. Mask the prompt — train on responses only",
    "",
    "Without this, the model also fits user/system tokens, which dilutes the signal. The",
    "instruction marker `<|im_start|>user\\n` and response marker `<|im_start|>assistant\\n` are",
    "Qwen3.5 (ChatML) specials.",
))

cells.append(code(
    "from unsloth.chat_templates import train_on_responses_only",
    "",
    "trainer = train_on_responses_only(",
    "    trainer,",
    "    instruction_part = '<|im_start|>user\\n',",
    "    response_part    = '<|im_start|>assistant\\n',",
    ")",
    "print('Loss masking applied: only assistant turns contribute to the loss.')",
))

cells.append(md(
    "## 11. Train",
    "",
    "Logs every 10 steps; checkpoints every 100 steps to `OUTPUT_DIR`, keeping the latest 2.",
))

cells.append(code(
    "import time",
    "t0 = time.time()",
    "stats = trainer.train()",
    "print(f'Training finished in {(time.time()-t0)/60:.1f} min')",
    "print(stats)",
))

cells.append(md(
    "## 12. Push LoRA adapter immediately",
    "",
    "Pushed before merging so a Colab disconnect during the merge step doesn't lose the adapter.",
))

cells.append(code(
    "model.push_to_hub(ADAPTER_REPO, token=HF_TOKEN, private=True)",
    "tokenizer.push_to_hub(ADAPTER_REPO, token=HF_TOKEN, private=True)",
    "print(f'Adapter pushed: https://huggingface.co/{ADAPTER_REPO}')",
))

cells.append(md(
    "## 13. Smoke-test the adapter",
    "",
    "Generates from one of the locked eval prompts (A-01: `errors.Join`) using the adapter still",
    "in memory. We sample with the Qwen3.5 thinking-mode decoding settings.",
))

cells.append(code(
    "FastLanguageModel.for_inference(model)",
    "",
    "messages = [",
    "    {'role': 'system', 'content': 'You are a Mantle coding assistant. Mantle is a Go-based model execution system centred on the Model Container Format (MCF). Follow AGENTS.md: stdlib-first Go, no stubs, no TODOs, modern Go, little-endian on-disk invariants, gofmt- and golangci-lint-clean.'},",
    "    {'role': 'user',   'content': 'In the Mantle codebase, a function validates an MCF header and today returns only the first error it finds. Rewrite it so that it runs all validation checks and returns a single joined error using stdlib. Do not introduce new dependencies. Keep behaviour deterministic. Provide the Go snippet and a two-sentence explanation.'},",
    "]",
    "prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)",
    "inputs = tokenizer(prompt, return_tensors='pt').to('cuda')",
    "out = model.generate(",
    "    **inputs,",
    "    max_new_tokens   = 800,",
    "    do_sample        = True,",
    "    temperature      = 0.6,",
    "    top_p            = 0.95,",
    "    top_k            = 20,",
    "    pad_token_id     = tokenizer.eos_token_id,",
    ")",
    "print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False))",
))

cells.append(md(
    "## 14. Merge LoRA into base, save bf16, push",
    "",
    "`save_method='merged_16bit'` writes a self-contained model directory we can serve without",
    "PEFT at runtime. GGUF export is intentionally skipped — that conversion is done downstream.",
))

cells.append(code(
    "model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method='merged_16bit')",
    "print('Merged checkpoint at:', MERGED_DIR)",
    "",
    "model.push_to_hub_merged(MERGED_REPO, tokenizer, save_method='merged_16bit', token=HF_TOKEN, private=True)",
    "print(f'Merged pushed:  https://huggingface.co/{MERGED_REPO}')",
))

cells.append(md(
    "## 15. Done",
    "",
    "- Adapter: `https://huggingface.co/{ADAPTER_REPO}` (private)",
    "- Merged:  `https://huggingface.co/{MERGED_REPO}` (private)",
    "",
    "Next: run the 30-prompt rubric in `scratchpad/mantle-finetune/eval/` against the merged",
    "checkpoint and record results in `notes/decisions.md`.",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

print(f"Wrote {OUT}")
print(f"Cells: {len(cells)} (markdown={sum(1 for c in cells if c['cell_type']=='markdown')}, code={sum(1 for c in cells if c['cell_type']=='code')})")
print(f"Size:  {OUT.stat().st_size:,} bytes")
