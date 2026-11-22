# -*- coding: utf-8 -*-
"""
Dataset builder: raw scripts -> SFT-ready JSONL with strict format enforcement.

Input : <root>/dataset/raw/*.txt   (one complete script per file)
Output: <root>/dataset/train.jsonl, <root>/dataset/val.jsonl
        <root>/dataset/rejected/  (failed scripts + reasons)

Run directly to rebuild (reads SAMMONIC_ROOT, defaults to the project root):

    python AI/datapipeline/build_dataset.py [--raw DIR]

Training format (completion-only SFT on the assistant turn):
{"messages": [
  {"role": "system",    "content": "<fixed system prompt: the format contract>"},
  {"role": "user",      "content": "<theme request>"},
  {"role": "assistant", "content": "<chatterbox-format script>"}
]}

Long scripts are split into continuation samples so every sample fits a
2048-token training window:
- first sample: user asks for the script's opening, assistant writes section 1
- later samples: user says "continue", assistant writes next section
Every sample is lint-validated before use.
"""

import os
import re
import json
import random
import glob

# Paths. This file lives at AI/datapipeline/, two levels below the project
# root, so the root must be resolved upward (or overridden by the sidecar via
# SAMMONIC_ROOT). Deriving paths from this file's own directory would silently
# point dataset/raw at AI/datapipeline/dataset/raw and lose every script.
ROOT = os.environ.get("SAMMONIC_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "dataset", "raw")
OUT_DIR = os.path.join(ROOT, "dataset")
REJ_DIR = os.path.join(OUT_DIR, "rejected")
TRAIN_JSONL = os.path.join(OUT_DIR, "train.jsonl")
VAL_JSONL = os.path.join(OUT_DIR, "val.jsonl")
SEED_SCRIPT = os.path.join(ROOT, "script.txt")  # used if raw/ is empty

ALLOWED_TAGS = {"[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]",
                "[clear throat]", "[sniff]", "[groan]", "[shush]"}
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF\U0001F900-\U0001F9FF\U0000FE00-\U0000FE0F\U0000200D]+")

SYSTEM_PROMPT = """You are a hypnosis script writer for Chatterbox TTS.
Write ONLY speakable script text in this exact format:

- Paragraphs of 200 to 500 characters, never more than 1000.
- Separate each paragraph with a blank line.
- Use ellipses (three dots) for pauses, never brackets.
- Use ONLY these bracketed sound tags inline: [laugh], [chuckle], [sigh],
  [gasp], [cough], [clear throat], [sniff], [groan], [shush].
- NEVER use other bracketed words, SSML, emojis, asterisks, or stage directions.
- Use ALL CAPS for loud or emphatic words.
- Spell out numbers and symbols. Complete sentences only.
Write the full session arc: induction, deepening, narrative journey,
suggestions, then a waking close."""

MAX_CHUNK = 1000     # max chars per script paragraph in training samples
SECTION_CHARS = 5500  # chars per continuation sample (fits ~2048 tokens w/ prompt)


# ----------------------------------------------------------------- linting
def lint_paragraphs(text, source=None):
    """Return list of problems for one script. Empty list means accept."""
    problems = []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) < 2:
        problems.append("fewer than 2 paragraphs")
    for i, p in enumerate(paras, 1):
        if len(p) > MAX_CHUNK:
            problems.append(f"para {i} over {MAX_CHUNK} chars ({len(p)})")
        if len(p) < 150:
            problems.append(f"para {i} under 150 chars ({len(p)})")
        if re.search(r"[a-z]\.[A-Z]", p):
            problems.append(f"para {i}: missing space after period")
        if re.search(r"\.\.\.\.", p):
            problems.append(f"para {i}: four+ dots")
    tags = re.findall(r"\[[a-z ]+\]", text)
    illegal = sorted(set(tags) - ALLOWED_TAGS)
    if illegal:
        problems.append("illegal tags: " + ", ".join(illegal))
    if not tags:
        problems.append("no sound tags at all (tagless scripts are allowed only if intentional)")
    if EMOJI_RE.search(text):
        problems.append("emoji/symbols present")
    if "<" in text and ">" in text:
        problems.append("angle brackets present")
    if re.search(r"\((\*|whisper|sigh|pause)", text, re.I):
        problems.append("stage directions present")
    return problems


def split_sections(text):
    """Split a script into sections of <= SECTION_CHARS on paragraph bounds."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    sections, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > SECTION_CHARS:
            sections.append(cur.strip())
            cur = p
        else:
            cur = (cur + "\n\n" + p).strip() if cur else p
    if cur:
        sections.append(cur.strip())
    return sections


def derive_theme(text):
    """Cheap theme guess from the script's opening, for the user prompt."""
    head = text[:400].lower()
    if "forest" in head or "tree" in head:
        return "a forest induction with a guide figure"
    if "escalator" in head or "stair" in head:
        return "an escalator descent induction"
    if "void" in head or "white space" in head:
        return "an infinite void induction"
    if "hell" in head or "demon" in head:
        return "an infernal descent with a demonic guide"
    if "swamp" in head or "mud" in head:
        return "a swamp/mud immersion journey"
    if "sleep" in head or "bed" in head:
        return "a sleep induction"
    return "an imaginative trance induction journey"


# -------------------------------------------------------------------- main
def build(raw_dir=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    if raw_dir is None:
        raw_dir = RAW_DIR
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(REJ_DIR, exist_ok=True)
    random.seed(42)

    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not raw_files and os.path.exists(SEED_SCRIPT):
        shutil_copy_seed()
        raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.txt")))
    if not raw_files:
        print("No scripts found. Drop .txt files in", RAW_DIR)
        return

    samples, rejected = [], []
    for idx, path in enumerate(raw_files, 1):
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        name = os.path.splitext(os.path.basename(path))[0]
        print("@@PROGRESS " + json.dumps(
            {"i": idx, "total": len(raw_files), "name": name}), flush=True)
        problems = lint_paragraphs(text, name)
        if problems:
            rejected.append((name, problems))
            with open(os.path.join(REJ_DIR, name + ".txt"), "w", encoding="utf-8") as f:
                f.write(text)
            with open(os.path.join(REJ_DIR, name + ".problems.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(problems))
            continue
        sections = split_sections(text)
        theme = derive_theme(text)
        for i, sec in enumerate(sections):
            if i == 0:
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Write a full hypnosis script with {theme}."},
                    {"role": "assistant", "content": sec},
                ]
            else:
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Continue the same script with the next section."},
                    {"role": "assistant", "content": sec},
                ]
            samples.append({"source": name, "part": i + 1, "messages": msgs})

    print(f"Scripts: {len(raw_files)} | rejected: {len(rejected)} | samples: {len(samples)}")

    random.shuffle(samples)
    val_n = max(1, int(len(samples) * 0.1)) if len(samples) >= 10 else 0
    val, train = samples[:val_n], samples[val_n:]
    for path, rows in ((TRAIN_JSONL, train), (VAL_JSONL, val)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    print(f"train: {len(train)} -> {TRAIN_JSONL}")
    print(f"val:   {len(val)} -> {VAL_JSONL}")
    if rejected:
        print("Rejected (see dataset/rejected/):")
        for name, probs in rejected:
            print(f"  {name}: {probs[:3]}")

    # Machine-readable summary for the sidecar, same protocol as generate.py.
    print("@@RESULT " + json.dumps({
        "scripts": len(raw_files),
        "accepted": len(raw_files) - len(rejected),
        "rejected": len(rejected),
        "samples": len(samples),
        "train": len(train),
        "val": len(val),
    }), flush=True)


def shutil_copy_seed():
    import shutil
    dst = os.path.join(RAW_DIR, "seed_session.txt")
    shutil.copy(SEED_SCRIPT, dst)
    print(f"Seeded raw set with {dst}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build SFT JSONL from raw scripts.")
    ap.add_argument("--raw", default=None,
                   help="raw script directory (default <root>/dataset/raw)")
    ap.add_argument("--root", default=None,
                   help="project root override (default: SAMMONIC_ROOT or inferred)")
    args = ap.parse_args()

    if args.root:
        # Re-derive every path under the override rather than mutating module
        # globals piecemeal, so an override can never leave half the paths stale.
        root = os.path.abspath(args.root)
        os.environ["SAMMONIC_ROOT"] = root
        globals()["ROOT"] = root
        globals()["RAW_DIR"] = os.path.join(root, "dataset", "raw")
        globals()["OUT_DIR"] = os.path.join(root, "dataset")
        globals()["REJ_DIR"] = os.path.join(root, "dataset", "rejected")
        globals()["TRAIN_JSONL"] = os.path.join(root, "dataset", "train.jsonl")
        globals()["VAL_JSONL"] = os.path.join(root, "dataset", "val.jsonl")
        globals()["SEED_SCRIPT"] = os.path.join(root, "script.txt")

    build(args.raw)