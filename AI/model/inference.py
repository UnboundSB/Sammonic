# -*- coding: utf-8 -*-
"""
Script-writer inference: prompt the fine-tuned model and lint its output.

    python AI/model/inference.py --theme "a swamp immersion journey"
    python AI/model/inference.py --theme "..." --out draft.txt
    python AI/model/inference.py --dry-run

Two rules this file enforces:

1. **The format contract is imported, never retyped.** `SYSTEM_PROMPT` comes
   from `AI.datapipeline.build_dataset`, the same place the dataset builder
   takes it from. A second copy would drift, and a model tuned to a drifted
   prompt produces scripts the linter then rejects.
2. **Generated text is linted before it is handed back.** Violations are
   reported, not silently returned: a script that breaks the TTS format
   produces audible glitches, and glitching at a listener is worse than
   failing visibly here.

Heavy imports live inside functions so `--dry-run` works on a machine without
a full inference stack.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

# This file is two directories below the project root (AI/model/inference.py),
# which is three .parent hops from the file path itself.
ROOT = Path(os.environ.get("SAMMONIC_ROOT") or
            str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(ROOT))

from AI.datapipeline.build_dataset import SYSTEM_PROMPT, lint_paragraphs  # noqa: E402

DEFAULT_ADAPTER = ROOT / "AI" / "model" / "adapters"
MERGED_DIR = ROOT / "AI" / "model" / "merged"
DEFAULT_CONFIG = ROOT / "AI" / "training" / "config.yaml"


def note(msg):
    print(f"[infer] {msg}", flush=True)


class InferenceError(RuntimeError):
    pass


# ------------------------------------------------------------------ checks
def resolve_adapter(explicit=None):
    """Prefer an explicitly passed adapter, then merged weights, then the
    newest single adapter directory. Pointing at a stale adapter silently
    produces yesterday's voice, so the choice is printed."""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise InferenceError(f"adapter not found: {p}")
        return p, "explicit"
    if MERGED_DIR.is_dir():
        candidates = [p for p in MERGED_DIR.iterdir()
                      if p.is_dir() and (p / "config.json").exists()]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0], "merged"
    if DEFAULT_ADAPTER.is_dir():
        candidates = [p for p in DEFAULT_ADAPTER.iterdir()
                      if p.is_dir() and (p / "adapter_config.json").exists()]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0], "lora adapter"
    raise InferenceError(
        "no adapter found: train first (AI/training/train_qlora.py) or pass "
        "--adapter to a checkpoint directory")


def base_model_id(config=DEFAULT_CONFIG):
    """Read the base model from the same config the trainer uses, so the
    adapter is loaded against the weights it was trained on."""
    if not Path(config).exists():
        raise InferenceError(f"config not found: {config}")
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    model = cfg.get("base_model")
    if not model:
        raise InferenceError("config.key 'base_model' is missing")
    return model


def dry_run():
    print("== DRY RUN (no weights loaded) ==")
    problems = []
    print(f"root          : {ROOT}")
    print(f"system prompt : imported from AI.datapipeline.build_dataset "
          f"({len(SYSTEM_PROMPT)} chars)")

    try:
        path, kind = resolve_adapter()
        print(f"adapter       : {path} ({kind})")
    except InferenceError as exc:
        problems.append(str(exc))
        print(f"adapter       : none ({exc})")

    try:
        model = base_model_id()
        print(f"base model    : {model}")
    except InferenceError as exc:
        problems.append(str(exc))
        print(f"base model    : unknown ({exc})")

    libs = []
    for mod, why in (("torch", "model execution"), ("transformers", "model + tokenizer"),
                    ("peft", "LoRA adapter loading")):
        try:
            __import__(mod)
        except ImportError:
            libs.append(f"{mod} -- {why}")
    if libs:
        print("\nmissing libraries (install before generating):")
        for entry in libs:
            print("  -", entry)
    else:
        print("\nall inference libraries import cleanly")

    if problems:
        print("\n== DRY RUN: NOT READY ==")
        for p in problems:
            print("  -", p)
        return 1
    if libs:
        print("\n== DRY RUN: READY (adapter + base model resolve; install libraries to generate) ==")
    else:
        print("\n== DRY RUN: READY (adapter, base model, and libraries all resolve) ==")
    return 0


# ------------------------------------------------------------------ generate
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate a hypnosis script with the fine-tuned model.")
    ap.add_argument("--theme", default="an imaginative trance induction journey",
                    help="theme request passed to the model")
    ap.add_argument("--out", default=None, help="write the script here")
    ap.add_argument("--adapter", default=None,
                    help="adapter or merged-weights directory (default: newest)")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify adapter and library availability without "
                         "loading weights or touching the GPU")
    ap.add_argument("--max-new-tokens", type=int, default=2200)
    args = ap.parse_args(argv)

    try:
        if args.dry_run:
            note(f"root: {ROOT}")
            return dry_run()
        generate(args.theme, args.adapter, args.out, args.max_new_tokens)
        return 0
    except InferenceError as exc:
        print(f"\nINFERENCE REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr, flush=True)
        return 130


def generate(theme, adapter_dir=None, out=None, max_new_tokens=2200,
             temperature=0.8, top_p=0.9, repetition_penalty=1.05):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise InferenceError(f"missing inference dependency: {exc}; install peft") from exc

    path, kind = resolve_adapter(adapter_dir)
    model_id = base_model_id()
    note(f"base model: {model_id}")
    note(f"adapter: {path} ({kind})")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    # A LoRA adapter is applied on top; merged weights are used directly.
    if (path / "adapter_config.json").exists():
        note("applying LoRA adapter to the base model")
        model = PeftModel.from_pretrained(model, str(path))
    else:
        note("loading merged weights directly")
    model.eval()

    user = f"Write a full hypnosis script with {theme}."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    note(f"prompt: {user}")
    note("sampling (this is the part that needs the GPU)")
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()

    problems = [p for p in lint_paragraphs(generated)
               if not p.startswith(("para ", "no sound tags"))]
    report = {
        "theme": theme,
        "base_model": model_id,
        "adapter": str(path),
        "chars": len(generated),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "linter": problems,
    }

    if out:
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(generated)
        report["out"] = str(out_path)
        note(f"wrote {out_path}")
    else:
        print("\n--- generated ---\n" + generated + "\n--- end ---")

    # Report violations instead of quietly passing them on: a format break
    # becomes an audible glitch in the TTS render.
    if problems:
        print("\nLINTER FAILURES in the generated script:", flush=True)
        for p in problems:
            print("  -", p)
        note("the script is still returned; fix it or regenerate")
    else:
        note("generated script passes the linter")

    log_dir = ROOT / "AI" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "inference.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")
    note(f"report appended to {log_dir / 'inference.log'}")
    return generated


if __name__ == "__main__":
    sys.exit(main())