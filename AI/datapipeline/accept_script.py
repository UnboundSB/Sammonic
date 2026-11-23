# -*- coding: utf-8 -*-
"""
File one candidate script through the linter, and only then into the corpus.

    python AI/datapipeline/accept_script.py draft.txt
    python AI/datapipeline/accept_script.py draft.txt --dry-run
    python AI/datapipeline/accept_script.py --status

This exists because the training gate needs 30 accepted scripts and the
corpus comes from a generation loop, not from a git checkout. Each draft has
to be judged somewhere, and the honest place is here: the same
`lint_paragraphs` that the dataset builder and the sidecar use, so "accepted"
means one thing everywhere.

A rejected draft is never written to dataset/raw. It is copied to
dataset/rejected/ with its reasons, which is the only way to tell later
whether a script was bad or just got lost.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SAMMONIC_ROOT") or str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(ROOT))

from AI.datapipeline.build_dataset import MAX_CHUNK, REJ_DIR, lint_paragraphs  # noqa: E402

# build_dataset exports paths as plain strings; wrap them so the same
# Path API works here.
RAW_DIR = ROOT / "dataset" / "raw"
REJ = Path(REJ_DIR) if REJ_DIR else ROOT / "dataset" / "rejected"


def read_candidate(path):
    p = Path(path)
    if not p.is_dir() and not p.exists():
        raise SystemExit(f"candidate not found: {p}")
    return p


def classify(problems):
    """Separate problems that block acceptance from ones worth knowing about."""
    blocking, advisory = [], []
    for p in problems:
        if p.startswith(("para ", "no sound tags")):
            advisory.append(p)
        else:
            blocking.append(p)
    return blocking, advisory


def accept(path, dry_run=False, tag=""):
    path = read_candidate(path)
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chars = len(text)
    blocking, advisory = classify(lint_paragraphs(text))

    print(f"candidate     : {path}")
    print(f"paragraphs    : {len(paras)}")
    print(f"chars         : {chars}")
    sizes = [len(p) for p in paras]
    if sizes:
        print(f"para sizes    : {min(sizes)}-{max(sizes)} (target 200-500, hard max {MAX_CHUNK})")
    if advisory:
        print("\nadvisory (does not block):")
        for a in advisory:
            print("  -", a)

    if blocking:
        print("\nBLOCKED by the linter:")
        for b in blocking:
            print("  -", b)
        if not dry_run:
            REJ.mkdir(parents=True, exist_ok=True)
            stem = path.stem or "candidate"
            stamp = time.strftime("%H%M%S")
            (REJ_DIR / f"{stem}.rejected.txt").write_text(text, encoding="utf-8")
            (REJ_DIR / f"{stem}.rejected.txt.problems").write_text(
                "\n".join(blocking + advisory), encoding="utf-8")
            print(f"\nquarantined to {REJ / (stem + '.rejected.txt')} with reasons")
        else:
            print("\n(dry run: nothing written)")
        return 2

    if dry_run:
        print("\n(dry run: would be accepted, nothing written)")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{tag + '_' if tag else ''}{int(time.time())}.txt"
    # A name collision would silently overwrite an earlier script, so keep
    # advancing rather than trusting one timestamp.
    while (RAW_DIR / name).exists():
        time.sleep(1.1)
        name = f"{tag + '_' if tag else ''}{int(time.time())}.txt"
    (RAW_DIR / name).write_text(text + "\n", encoding="utf-8")
    print(f"\naccepted -> {RAW_DIR / name}")
    print("run: python AI/datapipeline/build_dataset.py   to rebuild the dataset")
    return 0


def status():
    raw = sorted(RAW_DIR.glob("*.txt")) if RAW_DIR.is_dir() else []
    rej = sorted(REJ.glob("*.rejected.txt")) if REJ.is_dir() else []
    train = ROOT / "dataset" / "train.jsonl"
    print(f"raw scripts   : {len(raw)} (gate is 30)")
    for p in raw:
        print("   ", p.name)
    print(f"rejected      : {len(rej)}")
    print(f"train.jsonl   : {'present' if train.exists() else 'absent'}")
    if len(raw) < 30:
        print(f"\n{30 - len(raw)} more script(s) needed before training will start")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Lint one candidate script and file it into the corpus "
                    "only if it passes.")
    ap.add_argument("path", nargs="?", help="candidate .txt file")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen without writing anything")
    ap.add_argument("--tag", default="",
                    help="prefix for the accepted filename")
    ap.add_argument("--status", action="store_true",
                    help="show corpus counts against the 30-script gate")
    args = ap.parse_args(argv)

    if args.status or not args.path:
        return status()
    return accept(args.path, args.dry_run, args.tag)


if __name__ == "__main__":
    sys.exit(main())