# Sammonic

Local-first toolkit for building long-form hypnosis sessions with neural TTS —
script format enforcement, dataset contraction, QLoRA training config, and a
C++ piece-table editing core.

**No story content lives in this repo.** Scripts, audio, and logs are git-ignored
by design; this repository holds only tooling.

## Components

### 1. Chatterbox TTS pipeline (`generate.py`)
Paragraph-chunked generation for Chatterbox-Turbo, LUFS-matched seams, silence
trimming, peak guard, and MP3 export via ffmpeg. Reads `script.txt` + a
reference WAV clip; writes `output.wav` / `output.mp3`.

### 2. Format linter (in `build_dataset.py`)
Enforces the TTS-safe format contract on every script:
- 200–500 char paragraphs (1,000 hard max)
- only the nine paralinguistic tags: `[laugh] [chuckle] [sigh] [gasp] [cough] [clear throat] [sniff] [groan] [shush]`
- ellipses for pauses (never brackets), no SSML, no emojis, no stage directions
- violators quarantined to `dataset/rejected/` with reasons

### 3. Dataset builder (`build_dataset.py`)
`dataset/raw/*.txt` → SFT JSONL (`train.jsonl`, `val.jsonl`) with long scripts
split into continuation samples. Schema + sourcing plan in `dataset/SCHEMA.md`.

### 4. Session builder GUI (`hypno_maker.py`)
Tkinter front-end: reference-audio manager with duration validation, script
editor, one-click lint, threaded generation with per-chunk progress.

### 5. Piece-table editing core (`app/cpp/`)
Append-only-buffer piece table in C++ (C ABI bridge) for O(log n) edits on
multi-megabyte scripts with undo/redo — the planned heart of the desktop app's
editing surface.

### 6. Quick writer (`writer.py`)
Minimal append-loop for drafting scripts line-by-line into `script.txt`
(Ctrl+C to stop).

## Layout

```
generate.py            TTS generation pipeline
build_dataset.py       linter + dataset builder
hypno_maker.py         session builder GUI
writer.py              line-by-line script drafting
script.txt             (git-ignored) your current script
reference_audio.wav    (git-ignored) voice clone reference
output.mp3             (git-ignored) generated session
dataset/SCHEMA.md      dataset contract + sourcing plan
app/cpp/               piece-table engine
```

## Requirements

- Python 3.11+, `chatterbox-tts`, torch (CUDA recommended), `pyloudnorm`
- ffmpeg on PATH
- C++17 compiler (for the piece-table core)

## Status

Baseline: linter, dataset contract, TTS pipeline, GUI, and editing core are
implemented. QLoRA training config and the Lit/Tauri shell are next.