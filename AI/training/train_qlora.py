# -*- coding: utf-8 -*-
"""
4-bit QLoRA fine-tuning for the Sammonic script-writer.

    python AI/training/train_qlora.py --config AI/training/config.yaml
    python AI/training/train_qlora.py --dry-run

Two properties this file exists to guarantee:

1. **`--dry-run` is real.** The heavy imports (torch, transformers, peft,
   bitsandbytes, trl) happen inside functions, so on a machine where they are
   missing the dry run still verifies the config, the dataset shape, every
   assistant turn against the linter, the dataset gate, and the output paths,
   and reports exactly which libraries a real run would need. A training
   script you cannot check before spending an hour of GPU is a trap.
2. **The dataset gate is enforced here, before the GPU is touched.** Below
   `min_scripts`, the run refuses. The point is not tokens; it is that a
   handful of scripts gets memorized instead of generalized.

Progress is emitted as `@@PROGRESS {...}` / `@@RESULT {...}` lines, which
`AI/sidecar_server.py` parses. Human-readable output stays in the plain
print() lines.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import yaml

# This file is three levels below the project root (AI/training/train_qlora.py),
# so the root is three parents up -- one more than the sidecar or the ctypes
# bridge, which sit directly under AI/.
ROOT = Path(os.environ.get("SAMMONIC_ROOT") or
            str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(ROOT))

# The linter is imported eagerly but is pure stdlib, so it is safe here and
# the dry run can use it to audit the dataset before any heavy import.
from AI.datapipeline.build_dataset import SYSTEM_PROMPT, lint_paragraphs  # noqa: E402

DEFAULT_CONFIG = ROOT / "AI" / "training" / "config.yaml"


# ------------------------------------------------------------------ output
def emit_progress(payload):
    print("@@PROGRESS " + json.dumps(payload), flush=True)


def emit_result(payload):
    print("@@RESULT " + json.dumps(payload), flush=True)


def note(message):
    print(f"[train] {message}", flush=True)


class TrainingError(RuntimeError):
    """A problem worth failing loudly over, with a message a human can act on."""


# ------------------------------------------------------------------ config
def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise TrainingError(f"config not found: {path}")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TrainingError(f"config is not a mapping: {path}")
    required = ("base_model", "dataset_dir", "min_scripts",
               "max_seq_length", "lora_r", "lora_alpha",
               "per_device_train_batch_size", "gradient_accumulation_steps")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise TrainingError(f"config is missing required keys: {', '.join(missing)}")
    return cfg


def resolve(cfg, key, default=None):
    value = cfg.get(key, default)
    if value is None:
        raise TrainingError(f"config key '{key}' has no value")
    return value


def dataset_paths(cfg):
    ds = Path(resolve(cfg, "dataset_dir"))
    if not ds.is_absolute():
        ds = ROOT / ds
    return ds / resolve(cfg, "train_file", "train.jsonl"), \
           ds / resolve(cfg, "val_file", "val.jsonl"), ds


# ------------------------------------------------------------------ dataset
def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise TrainingError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
    return rows


def dataset_digest(paths):
    """Stable hash of the training data, recorded in run_meta.json so two runs
    on different data are distinguishable after the fact."""
    h = hashlib.sha256()
    for p in sorted(paths):
        if not p.exists():
            continue
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def audit_dataset(rows):
    """Shape and content audit, independent of the training stack.

    Returns (samples, problems). Problems are things that must stop a real
    run: an assistant turn that the linter rejects, or a turn that is
    obviously truncated, means the dataset builder and this script disagree
    and training on it would teach the model to violate its own contract.
    """
    samples, problems = 0, []
    for i, row in enumerate(rows):
        msgs = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(msgs, list) or not msgs:
            problems.append(f"sample {i}: no messages array")
            continue
        roles = [m.get("role") for m in msgs if isinstance(m, dict)]
        if "assistant" not in roles:
            problems.append(f"sample {i}: no assistant turn")
            continue
        if "system" not in roles:
            problems.append(f"sample {i}: no system turn (format contract missing)")
        text = next(m["content"] for m in msgs
                    if isinstance(m, dict) and m.get("role") == "assistant")
        if not isinstance(text, str) or not text.strip():
            problems.append(f"sample {i}: empty assistant content")
            continue
        samples += 1
        lint = lint_paragraphs(text)
        hard = [p for p in lint if not p.startswith(("para ", "no sound tags"))]
        if hard:
            problems.append(f"sample {i}: assistant turn fails the linter: {hard[:3]}")
    return samples, problems


def count_source_scripts(ds_dir):
    raw = ds_dir / "raw"
    if not raw.is_dir():
        return 0
    return len([p for p in raw.glob("*.txt") if p.is_file()])


def check_gate(cfg, samples, source_scripts):
    gate_scripts = int(resolve(cfg, "min_scripts", 30))
    gate_samples = int(resolve(cfg, "min_samples", 0))
    reasons = []
    if source_scripts < gate_scripts:
        reasons.append(
            f"only {source_scripts} script(s) in dataset/raw, gate requires {gate_scripts}; "
            f"add {gate_scripts - source_scripts} more before training or the model "
            f"memorizes this handful instead of learning the format")
    if samples < gate_samples:
        reasons.append(f"{samples} sample(s) in train+val, gate requires {gate_samples}")
    return reasons


# ------------------------------------------------------------------ dry run
def missing_libraries():
    wanted = {
        "torch": "model loading and training loop",
        "transformers": "model and tokenizer",
        "peft": "LoRA adapters",
        "bitsandbytes": "4-bit quantization and paged optimizers",
        "datasets": "dataset streaming (only if train_file is not local JSONL)",
    }
    missing = []
    for mod, why in wanted.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(f"{mod} -- {why}")
    return missing


def dry_run(config_path):
    print("== DRY RUN (no GPU, no weights loaded) ==")
    problems = []
    cfg = load_config(config_path)
    print(f"config        : {config_path}")
    model = cfg.get("base_model", "")
    print(f"base_model    : {model}")

    if not model or "/" not in model:
        problems.append(f"base_model looks malformed: {model!r}")
    else:
        owner, name = model.split("/", 1)
        if not name.strip():
            problems.append("base_model has an empty model name after '/'")

    train_path, val_path, ds_dir = dataset_paths(cfg)
    total_samples, scripts = 0, 0
    for label, path in (("train", train_path), ("val", val_path)):
        if path.exists():
            rows = read_jsonl(path)
            n, probs = audit_dataset(rows)
            total_samples += n
            print(f"{label + ' file':14s}: {path.relative_to(ROOT)} -- {n} samples")
            problems.extend(f"{label}: {p}" for p in probs)
        else:
            print(f"{label + ' file':14s}: MISSING ({path})")
            problems.append(f"{label} file not found: {path}")

    scripts = count_source_scripts(ds_dir)
    print(f"raw scripts   : {scripts} (gate {cfg.get('min_scripts', 30)})")
    problems.extend(check_gate(cfg, total_samples, scripts))

    # Output paths must be writable or the run dies after hours of work.
    out = Path(resolve(cfg, "output_dir", "AI/model/adapters"))
    if not out.is_absolute():
        out = ROOT / out
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_test"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        print(f"adapter out   : {out} (writable)")
    except OSError as exc:
        problems.append(f"output dir not writable: {exc}")

    events = Path(resolve(cfg, "events_file", "AI/training/events.jsonl"))
    if not events.is_absolute():
        events = ROOT / events
    try:
        events.parent.mkdir(parents=True, exist_ok=True)
        print(f"events file   : {events}")
    except OSError as exc:
        problems.append(f"events dir not writable: {exc}")

    seq = int(resolve(cfg, "max_seq_length", 2048))
    bs = int(resolve(cfg, "per_device_train_batch_size", 1))
    acc = int(resolve(cfg, "gradient_accumulation_steps", 16))
    epochs = int(resolve(cfg, "num_train_epochs", 2))
    steps = max(1, total_samples) * epochs // max(1, bs * acc)
    print(f"plan          : {total_samples} samples x {epochs} epochs, "
          f"batch {bs}x{acc}, seq {seq} -> ~{steps} optimizer steps")

    libs = missing_libraries()
    if libs:
        print("\nmissing libraries (install before a real run):")
        for entry in libs:
            print("  -", entry)
    else:
        print("\nall training libraries import cleanly")

    if problems:
        print("\n== DRY RUN: NOT READY ==")
        for p in problems:
            print("  -", p)
        return 1

    if libs:
        print("\n== DRY RUN: READY (config + data OK; install libraries to train) ==")
    else:
        print("\n== DRY RUN: READY (config, data, and libraries all OK) ==")
    return 0


# ------------------------------------------------------------------ training
def train(config_path, resume=None):
    """The real run. Every heavy import lives here, so --dry-run stays usable
    on a machine without a working training stack."""
    cfg = load_config(config_path)
    train_path, val_path, ds_dir = dataset_paths(cfg)

    note(f"base model: {cfg.get('base_model')}")

    train_rows = read_jsonl(train_path) if train_path.exists() else []
    val_rows = read_jsonl(val_path) if val_path.exists() else []
    if not train_rows:
        raise TrainingError(f"no training data in {train_path}; run "
                            f"python AI/datapipeline/build_dataset.py first")

    samples, problems = audit_dataset(train_rows + val_rows)
    problems.extend(check_gate(cfg, samples, count_source_scripts(ds_dir)))
    if problems:
        raise TrainingError("dataset rejected:\n  " + "\n  ".join(problems))
    note(f"dataset accepted: {len(train_rows)} train, {len(val_rows)} val, "
         f"{count_source_scripts(ds_dir)} source scripts")

    # ---- imports the dry run does not need ---------------------------
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig, DataCollatorForSeq2Seq,
                                   Trainer, TrainingArguments)
    except ImportError as exc:
        raise TrainingError(
            f"missing training dependency: {exc}. Install peft, bitsandbytes, "
            f"accelerate, and datasets, then re-run.") from exc

    model_id = cfg.get("base_model")
    attn_impl = "sdpa" if torch.cuda.is_available() else "eager"

    note(f"loading {model_id} (4-bit NF4, attn={attn_impl})")
    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        # Without a pad token the batched collator cannot distinguish padding
        # from content, and the loss would be computed over padding positions.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float16,
        attn_implementation=attn_impl,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=int(resolve(cfg, "lora_r", 32)),
        lora_alpha=int(resolve(cfg, "lora_alpha", 64)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(resolve(cfg, "lora_target_modules",
                                    ["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])),
    )
    model = get_peft_model(model, lora)
    # The trainable-parameter report is the cheapest way to see that the LoRA
    # adapters are actually attached. Without it a mis-typed target_modules
    # silently trains nothing, or trains the entire frozen base.
    model.print_trainable_parameters()

    # ---- data ---------------------------------------------------------
    ds = Dataset.from_list(train_rows).train_test_split(
        test_size=0.1, seed=int(cfg.get("seed", 42))) if not val_rows else None
    if val_rows:
        train_ds = Dataset.from_list(train_rows)
        val_ds = Dataset.from_list(val_rows)
    else:
        train_ds, val_ds = ds["train"], ds["test"]

    run_name = cfg.get("run_name") or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(resolve(cfg, "output_dir", "AI/model/adapters"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = Path(resolve(cfg, "events_file", "AI/training/events.jsonl"))
    if not events_path.is_absolute():
        events_path = ROOT / events_path
    events_path.parent.mkdir(parents=True, exist_ok=True)

    bs = int(resolve(cfg, "per_device_train_batch_size", 1))
    acc = int(resolve(cfg, "gradient_accumulation_steps", 16))
    epochs = int(resolve(cfg, "num_train_epochs", 2))
    eff = bs * acc
    total_steps = max(1, len(train_ds) * epochs // max(1, eff))

    args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=acc,
        learning_rate=float(resolve(cfg, "learning_rate", 1.0e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        bf16=bool(cfg.get("bf16", True)),
        logging_steps=int(cfg.get("logging_steps", 5)),
        save_strategy=cfg.get("save_strategy", "epoch"),
        report_to="none",
        run_name=run_name,
        eval_strategy="epoch" if val_ds else "no",
        remove_unused_columns=False,
    )

    # Completion-only loss. Rather than trust a collator whose masking
    # conventions differ across library versions, render the conversation
    # with the model's own chat template, tokenize it once, then mask every
    # token outside the assistant span. Training on the instruction as well
    # teaches the model to re-emit instructions it is never asked for at
    # inference time, so this masking is a correctness requirement, not a
    # refinement.
    def tokenize_and_mask(row):
        max_len = int(resolve(cfg, "max_seq_length", 2048))
        msgs = [m for m in row["messages"] if isinstance(m, dict)]
        full = tokenizer.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=False)
        ids = tokenizer(full, add_special_tokens=False)["input_ids"]

        assistant_text = next((m.get("content", "") for m in msgs
                               if m.get("role") == "assistant"), None)
        labels = [-100] * len(ids)
        if assistant_text:
            a_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
            span = len(a_ids)
            n, m = len(ids), len(a_ids)
            # Locate the assistant tokens as a contiguous run inside the
            # full rendering; the chat template wraps them, so it is a
            # substring, not necessarily at the end.
            for start in range(n - m + 1):
                if ids[start:start + m] == a_ids:
                    labels[start:start + m] = a_ids
                    break
            else:
                # Fallback: mask the last `m` tokens, which is where an
                # assistant turn lands in every chat template in this family.
                if m <= n:
                    labels[n - m:n] = a_ids
        return {"input_ids": ids[:max_len], "labels": labels[:max_len]}

    note(f"tokenizing {len(train_ds)} samples (completion-only loss)")
    train_ds = train_ds.map(tokenize_and_mask)
    if val_path.exists():
        val_ds = val_ds.map(tokenize_and_mask)

    # Drop the raw text columns. The collator pads whatever columns remain, and a
    # list-of-dicts column cannot become a tensor -- it would fail deep inside
    # tokenizer.pad with a complaint about "excessive nesting" rather than
    # anything pointing at this line.
    drop_cols = [c for c in ("messages", "source", "part")
                 if c in train_ds.column_names]
    if drop_cols:
        train_ds = train_ds.remove_columns(drop_cols)
    if val_path.exists():
        vdrop = [c for c in ("messages", "source", "part")
                 if c in val_ds.column_names]
        if vdrop:
            val_ds = val_ds.remove_columns(vdrop)
    note(f"training columns: {train_ds.column_names}")

    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, pad_to_multiple_of=8,
        label_pad_token_id=-100, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if val_path.exists() else None,
        data_collator=collator,
    )

    note("training -- this is the part that needs the GPU")
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    meta = {
        "run_name": run_name,
        "base_model": model_id,
        "config": str(config_path),
        "samples": len(train_ds),
        "val_samples": len(val_ds) if val_path.exists() else 0,
        "source_scripts": count_source_scripts(ds_dir),
        "dataset_sha": dataset_digest([train_path, val_path]) if train_path.exists() else "",
        "hyperparameters": {
            "lora_r": int(resolve(cfg, "lora_r", 32)),
            "lora_alpha": int(resolve(cfg, "lora_alpha", 64)),
            "learning_rate": float(resolve(cfg, "learning_rate", 1.0e-4)),
            "epochs": epochs, "batch": bs, "accum": acc,
            "max_seq_length": int(resolve(cfg, "max_seq_length", 2048)),
        },
        "final_loss": result.training_loss,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "complete", **meta}) + "\n")

    emit_result({
        "adapter": str(run_dir),
        "final_loss": result.training_loss,
        "samples": len(train_ds),
        "config": str(config_path),
        "base_model": model_id,
    })
    note(f"adapter written to {run_dir}")
    return 0


# ------------------------------------------------------------------ entry
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="4-bit QLoRA fine-tuning for the Sammonic script-writer.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="path to config.yaml (default: AI/training/config.yaml)")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify config, dataset, gate, paths, and libraries "
                         "without loading weights or touching the GPU")
    ap.add_argument("--resume", default=None,
                    help="adapter directory (or checkpoint) to resume from")
    args = ap.parse_args(argv)

    try:
        if args.dry_run:
            note(f"root: {ROOT}")
            return dry_run(args.config)
        return train(args.config, args.resume)
    except TrainingError as exc:
        print(f"\nTRAINING REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())