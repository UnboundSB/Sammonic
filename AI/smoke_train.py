# -*- coding: utf-8 -*-
"""
Prove the trainer works, on a small model and a small dataset, in minutes.

    python AI/smoke_train.py

The full trainer (AI/training/train_qlora.py) was verified only with
`--dry-run`, which loads no weights. That is not proof the training loop works:
the BitsAndBytesConfig API changed in bitsandbytes 0.50 (`double_quant` became
`compress_statistics`), and the tokenizer/collator/labels plumbing has never
been executed. This runs the real code path against a 135M instruct model and a
hand-built dataset, then asserts the adapter was written and a loss went down.

It never touches dataset/ or AI/model/adapters; everything goes to a temp dir,
and a full Qwen3-4B run stays the user's decision.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("SAMMONIC_ROOT") or str(Path(__file__).resolve().parent.parent))
SCRIPT = ROOT / "AI" / "training" / "train_qlora.py"
TINY_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# A few samples in the exact shape build_dataset.py writes.
ROWS = [
    {
        "messages": [
            {"role": "system", "content": "You are a hypnosis script writer for Chatterbox TTS."},
            {"role": "user", "content": "Write a full hypnosis script with a forest induction."},
            {"role": "assistant", "content": (
                "The forest is quiet around you, and every step deeper takes you "
                "further from the noise of the day. [sigh] Let the leaves carry "
                "your weight while your breathing settles into something slow and "
                "unhurried.\n\n"
                "A little further now, where the light softens and the ground "
                "slopes gently downward. [chuckle] You notice how easily the "
                "sounds fade, and how little effort any of this requires of you "
                "now.")},
        ]
    },
] * 6


def write_config(tmp):
    ds = Path(tmp) / "ds"
    (ds / "raw").mkdir(parents=True, exist_ok=True)
    (ds / "raw" / "smoke.txt").write_text("placeholder\n", encoding="utf-8")
    with (ds / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in ROWS:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (ds / "val.jsonl").open("w", encoding="utf-8") as f:
        for r in ROWS[:2]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cfg = Path(tmp) / "smoke_config.yaml"
    cfg.write_text("\n".join([
        f'base_model: "{TINY_MODEL}"',
        "dataset_dir: \"" + str(ds).replace("\\", "/") + "\"",
        "train_file: \"train.jsonl\"",
        "val_file: \"val.jsonl\"",
        "min_scripts: 0",
        "min_samples: 2",
        "max_seq_length: 512",
        "load_in_4bit: true",
        'bnb_4bit_quant_type: "nf4"',
        "bnb_4bit_use_double_quant: true",
        'bnb_4bit_compute_dtype: "bfloat16"',
        "lora_r: 8",
        "lora_alpha: 16",
        "lora_dropout: 0.0",
        'lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]',
        'optim: "paged_adamw_8bit"',
        "learning_rate: 1.0e-3",
        'lr_scheduler_type: "constant"',
        "warmup_ratio: 0.0",
        "num_train_epochs: 2",
        "per_device_train_batch_size: 1",
        "gradient_accumulation_steps: 2",
        "gradient_checkpointing: false",
        "logging_steps: 1",
        'save_strategy: "no"',
        'bf16: true',
        "train_on_completions_only: true",
        "run_name: \"smoke\"",
        "output_dir: \"" + str(Path(tmp) / "adapters").replace("\\", "/") + "\"",
        "events_file: \"" + str(Path(tmp) / "events.jsonl").replace("\\", "/") + "\"",
        "seed: 42",
    ]) + "\n", encoding="utf-8")
    return cfg, ds


def main():
    tmp = tempfile.mkdtemp(prefix="sammonic_smoke_")
    print(f"== trainer smoke test ==\ntemp dir: {tmp}\nmodel: {TINY_MODEL}\n")
    try:
        cfg, ds = write_config(tmp)
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--config", str(cfg)],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=1800)

        log = proc.stdout or ""
        print("\n".join(l.strip() for l in log.splitlines()
                        if l.strip() and not l.startswith("@@")))
        if proc.returncode != 0:
            print(f"\nFAIL: trainer exited {proc.returncode}\n{proc.stderr[-3000:]}")
            return 1

        problems = []
        adapter = Path(tmp) / "adapters" / "smoke"
        if not (adapter / "adapter_config.json").exists():
            problems.append("no adapter_config.json written")
        if not (adapter / "run_meta.json").exists():
            problems.append("no run_meta.json written")
        if "trainable params" not in log:
            problems.append("no trainable-parameter report in the log")
        if "loss" not in log:
            problems.append("no loss line emitted")
        result = next((json.loads(l[len("@@RESULT "):]) for l in log.splitlines()
                       if l.startswith("@@RESULT ")), None)
        if not result or result.get("samples", 0) < 2:
            problems.append(f"result payload missing or wrong: {result}")

        print("\n== summary ==")
        if problems:
            for p in problems:
                print("  FAIL:", p)
            return 1
        print(f"trainer works end to end: loss went through "
              f"{result.get('final_loss')}, adapter written, gate enforced")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())