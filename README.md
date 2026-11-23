# Sammonic

Local-first toolkit for building long-form hypnosis sessions with neural TTS —
script format enforcement, dataset contraction, QLoRA training config, and a
C++ piece-table editing core.

**No story content lives in this repo.** Scripts, audio, logs, checkpoints,
`node_modules` and build output are git-ignored by design; this repository
holds only tooling.

## Components

### 1. Chatterbox TTS pipeline (`generate.py`)
Paragraph-chunked generation for Chatterbox-Turbo, LUFS-matched seams, silence
trimming, peak guard, and MP3 export via ffmpeg. Reads `script.txt` + a
reference WAV clip; writes `output.wav` / `output.mp3`. Emits
`@@PROGRESS` / `@@RESULT` heartbeat lines so progress is machine-readable.

### 2. Format linter (`AI/datapipeline/build_dataset.py`)
Enforces the TTS-safe format contract on every script:
- 200–500 char paragraphs (1,000 hard max)
- only the nine paralinguistic tags: `[laugh] [chuckle] [sigh] [gasp] [cough] [clear throat] [sniff] [groan] [shush]`
- ellipses for pauses (never brackets), no SSML, no emojis, no stage directions
- violators quarantined to `dataset/rejected/` with reasons

`SYSTEM_PROMPT` in this file is the single source of the format contract — the
dataset builder, the sidecar's `/lint`, and the inference harness all import it
rather than keeping copies that could drift.

### 3. Dataset builder (`AI/datapipeline/build_dataset.py`)
`dataset/raw/*.txt` → SFT JSONL (`train.jsonl`, `val.jsonl`) with long scripts
split into continuation samples. Schema + sourcing plan in `dataset/SCHEMA.md`.
Paths resolve from `SAMMONIC_ROOT` or the project root, so it runs from
anywhere.

### 4. Session builder GUI (`hypno_maker.py`)
Tkinter front-end: reference-audio manager with duration validation, script
editor, one-click lint, threaded generation with per-chunk progress. This is
the interim UI; the Lit frontend below is the real one.

### 5. Piece-table editing core (`app/cpp/`)
Append-only-buffer piece table in C++ with a C ABI, for O(log n) edits on
multi-megabyte scripts with undo/redo. Two test binaries cover the engine
(233 checks) and the ABI (46 checks, including null handles and 64 concurrent
handles). Built with CMake:

```
cmake -S app/cpp -B app/cpp/build && cmake --build app/cpp/build
```

MinGW builds statically link their runtimes, because a dynamically-linked DLL
carries hidden dependencies on the MSYS2 runtimes and fails to load on a
machine without them.

### 6. Sidecar service (`AI/sidecar_server.py`)
A stdlib-only HTTP server that fronts every heavy operation. Long jobs run as
child processes so torch is never loaded by the server and a crashed job cannot
take it down. The GPU is serialized: `train` and `generate` both need it, and a
request arriving while one runs gets `409` naming the job holding it rather than
sitting in a queue the caller cannot see.

| Route | Purpose |
|---|---|
| `POST /lint` | format gate (in-process, pure stdlib) |
| `POST /dataset/build` | rebuild the SFT JSONL |
| `POST /train` | start a QLoRA run |
| `POST /generate` | render a session to MP3 |
| `POST /edit/open` … `/edit/close` | editing against the C++ core |
| `GET /status/:id`, `POST /job/:id/cancel`, `GET /jobs` | job control |

The editing core is reachable from Python through stdlib `ctypes`
(`AI/piecetable_client.py`), so the UI needs no bridge process and no Node
addon. **All offsets are byte offsets**, and the header's "chars" wording is
misleading — the engine is byte-correct, as its UTF-8 tests prove.

### 7. Frontend (`app/web/`)
Lit 3 + Vite, five panels (script editor bound to the piece table, lint,
dataset, train, generate). One origin: it talks only to the sidecar.

```
cd app/web && npm install && npm run dev
```

### 8. Training and inference
- `AI/training/config.yaml` — every hyperparameter, commented with its reason.
  Sized for a 6 GB laptop GPU.
- `AI/training/train_qlora.py` — 4-bit QLoRA, completion-only loss, and a
  dataset gate that refuses to train below 30 accepted scripts.
- `AI/model/inference.py` — loads base + adapter, samples, and lints its own
  output before handing it back.

Both trainers have a `--dry-run` that verifies config, dataset, the gate, and
paths without loading weights, so a broken config is caught before an hour of
GPU time is spent.

## Layout

```
generate.py            TTS generation pipeline
hypno_maker.py         session builder GUI (interim UI)
writer.py              line-by-line script drafting
script.txt             (git-ignored) your current script
reference_audio.wav    (git-ignored) voice clone reference
output.mp3             (git-ignored) generated session
verify.py              run everything that can be checked without a GPU
architecture.md        the full system design
dataset/SCHEMA.md      dataset contract + sourcing plan
app/cpp/               piece-table engine + C ABI + tests
app/web/               Lit frontend
AI/                    sidecar, dataset pipeline, training, model
```

## Requirements

- Python 3.11+, `chatterbox-tts`, torch (CUDA recommended), `pyloudnorm`, `pyyaml`
- `peft`, `bitsandbytes`, `accelerate` — required only for a real training run
- ffmpeg on PATH
- C++17 compiler and CMake (for the piece-table core)
- Node + npm (only for the Lit frontend)

## Verify it works

```
python verify.py                      # fast: syntax, C++ tests, sidecar self-test
python AI/integration_generate.py     # slow: real TTS render through the sidecar
```

`verify.py` runs Python syntax checks, both C++ test binaries (building them if
needed), and the full sidecar self-test against a live server on a scratch port.
It deliberately does *not* run a TTS render or a training run.

`integration_generate.py` exercises the one seam the fast tests cannot: it starts
a real job through the sidecar and confirms the parsed `@@RESULT` duration
against `ffprobe`. It writes to `AI/logs/`, never the project's `output.mp3`.

## Status

Working and verified: TTS pipeline, linter, dataset builder, piece-table core
and ABI, ctypes bridge, sidecar service, and the Lit frontend. A full session has
been rendered end to end *through* the sidecar, so every layer between a
click and an MP3 has been executed at least once.

Still open: a real QLoRA run (needs `peft`, `bitsandbytes`, `accelerate`), enough
scripts to clear the 30-script training gate, browser automation for the frontend,
and the Tauri shell (no Rust toolchain is installed on the target machine).