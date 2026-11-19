# Sammonic — System Architecture

Local-first, RAM-disciplined toolkit for authoring long-form hypnosis sessions,
curating training corpora, fine-tuning a script-writing LM, and rendering
finished sessions to audio with neural TTS.

No story content in this document or this repository — engineering only.

---

## 1. Design Principles

| Principle | How it is enforced |
|---|---|
| **Local-first** | Every stage runs on-device. No cloud APIs, no telemetry, no external calls. |
| **RAM/VRAM discipline** | Editing is O(log n) append-only (C++); training is 4-bit QLoRA sized for a 6 GB laptop GPU; generation is chunked. |
| **Separation of concerns** | Text editing, dataset work, training, inference, and audio are independent processes that talk over narrow contracts. |
| **Fail loudly at the boundary** | The format linter is a hard gate: anything that fails it never reaches a dataset or the TTS engine. |
| **Incremental replaceability** | Each service can be swapped (e.g. Tkinter GUI → Lit/Tauri) without touching the others. |

---

## 2. System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                               │
│  ┌───────────────────────┐   ┌──────────────────────────────────┐   │
│  │  Lit 3 web components│   │  Tauri 2 shell (Rust)            │   │
│  │  - ScriptEditor       │   │  - window, menus, file dialogs   │   │
│  │  - LintPanel          │   │  - sidecar lifecycle             │   │
│  │  - DatasetPanel       │   │  - C++ core via FFI              │   │
│  │  - TrainPanel         │   │                                  │   │
│  │  - GeneratePanel      │   │                                  │   │
│  └──────────┬────────────┘   └───────────────┬──────────────────┘   │
│             │  fetch() / custom events       │  tauri::command      │
└─────────────┼───────────────────────────────┼──────────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────────┐   ┌───────────────────────────────────┐
│   BRIDGE (dev mode)         │   │   C++ EDITING CORE                │
│   Node + koffi FFI          │   │   piece-table engine (C ABI)      │
│   exposes HTTP :8765        │   │   - O(log n) insert/erase         │
│   - /edit/insert            │   │   - undo/redo stacks              │
│   - /edit/erase             │   │   - flat RAM via append-only      │
│   - /edit/undo /redo        │   │     original/add buffers          │
│   - /edit/text (full)       │   │   - DLL: piecetable.dll           │
│   - /edit/length            │   │                                   │
└─────────────┬───────────────┘   └───────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        PYTHON SERVICE LAYER                         │
│  sidecar HTTP server (stdlib http.server, no dependencies)          │
│                                                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐ ┌────────────┐ │
│  │ /lint        │ │ /dataset     │ │ /train        │ │ /generate  │ │
│  │ format gate  │ │ build JSONL  │ │ QLoRA runner │ │ TTS audio  │ │
│  └──────┬───────┘ └──────┬───────┘ └───────┬───────┘ └─────┬──────┘ │
└─────────┼────────────────┼─────────────────┼───────────────┼────────┘
          ▼                ▼                 ▼               ▼
   script.txt      dataset/*.jsonl    AI/model/        output.mp3
                   dataset/raw/       adapters
                   dataset/rejected/
```

---

## 3. Component Detail

### 3.1 Frontend — Lit 3 + Tauri 2

**Shell (Tauri, Rust):**
- Window management, native menus, file save/open dialogs.
- Launches the Python sidecar as a child process; pipes its stdout/stderr to the log view; kills it on exit.
- Loads `piecetable.dll` via `extern "C"` FFI and exposes `insert`, `erase`, `undo`, `redo`, `get_text`, `length` as Tauri commands — this is the production path for the editing core.
- Capabilities: `fs:default`, `shell:allow-spawn` scoped to the sidecar binary, `http:default` scoped to `http://127.0.0.1:8765/*` for dev-mode fallback.

**UI (Lit 3, TypeScript):**
- `script-editor` — virtualized textarea bound to the piece table (dev mode: HTTP bridge; prod mode: Tauri commands). Undo/redo buttons wired to the core.
- `lint-panel` — POSTs the script to `/lint`, renders errors/warnings/info from the shared Python linter (single source of truth).
- `dataset-panel` — lists `dataset/raw/`, runs `/dataset/build`, shows accepted/rejected counts with reasons.
- `train-panel` — starts `/train`, streams progress logs, shows loss/step from the trainer's JSONL events.
- `generate-panel` — reference-audio picker with duration probe, chunk progress bar, final MP3 duration on completion.
- `state` — a tiny store (no Redux) holding: current script, lint result, job status, logs.

**Dev mode:** `npm run dev` (Vite) serves the Lit app; a `bridge.mjs` (Node + koffi) stands in for the Tauri FFI layer so the UI runs before the Rust toolchain exists.

### 3.2 C++ Editing Core

- **File:** `app/cpp/piece_table.{h,cpp}` — piece table with append-only `orig`/`add` buffers, coalescing on the ADD hot path, undo/redo via forward-edit op stacks.
- **Why C++:** edits on a multi-MB script stay flat in RAM (never materialize the full string except for `slice`), and undo history costs nothing extra.
- **C ABI** (`c_api.cpp`): opaque handle + `pt_create`, `pt_destroy`, `pt_insert`, `pt_erase`, `pt_undo`, `pt_redo`, `pt_length`, `pt_get_text`, `pt_free_string`.
- **Build:** CMake (MSVC in Tauri builds; g++ for standalone tests).

### 3.3 Python Service Layer

Single sidecar process (`AI/sidecar_server.py`, stdlib only) exposing four endpoints, each mapping to one job:

| Endpoint | Job | Script |
|---|---|---|
| `POST /lint` | Format gate — chunk sizes, tag legality, emoji, SSML, stage directions | linter inside `build_dataset.py` |
| `POST /dataset/build` | `dataset/raw/*.txt` → validated `train.jsonl`/`val.jsonl`; violators to `dataset/rejected/` with reasons | `build_dataset.py` |
| `POST /train` | QLoRA fine-tune run (async; streams progress) | `AI/training/train_qlora.py` |
| `POST /generate` | Chatterbox TTS session render (async; streams chunk progress) | `generate.py` |
| `GET /status/:id` | Job progress polling | — |

Long jobs (train/generate) run as background threads; the frontend polls `/status`.

---

## 4. Directory Map

```
D:\hypno maker\
├── architecture.md          ← this document
├── README.md                repo overview
├── .gitignore               blocks: scripts, audio, logs, venv, checkpoints
│
├── app/
│   ├── cpp/
│   │   ├── piece_table.h        piece-table engine
│   │   ├── piece_table.cpp      implementation
│   │   ├── c_api.cpp            extern "C" bridge (planned)
│   │   └── CMakeLists.txt       (planned)
│   └── web/                     Lit + Tauri shell (planned)
│       ├── package.json
│       ├── src/hypno-app.ts
│       └── src-tauri/
│
├── AI/
│   ├── datapipeline/            raw ingestion + validation + JSONL build
│   │   ├── build_dataset.py         (moved from repo root)
│   │   └── SCHEMA.md                (moved from dataset/)
│   ├── model/
│   │   ├── inference.py         script-writer adapter loading + sampling
│   │   ├── adapters/            LoRA checkpoint dirs (git-ignored)
│   │   └── merged/              merged full weights (git-ignored)
│   └── training/
│       ├── train_qlora.py       4-bit QLoRA training loop
│       ├── config.yaml          hyperparameters, paths
│       └── events.jsonl         live training telemetry (git-ignored)
│
├── sidecar helpers (repo root, current generation)
│   ├── generate.py             Chatterbox TTS pipeline
│   ├── hypno_maker.py          Tkinter GUI (interim UI)
│   ├── writer.py               line-by-line drafting loop
│   └── build_dataset.py        linter + dataset builder (dual-homed with AI/datapipeline)
│
└── dataset/
    ├── raw/                    source scripts (git-ignored — content)
    ├── train.jsonl / val.jsonl derived SFT data (git-ignored — content)
    └── rejected/               linter failures (git-ignored — content)
```

---

## 5. IPC Contracts

### 5.1 C++ core (C ABI)
```c
void* pt_create(const char* initial_text);
void  pt_destroy(void* handle);
void  pt_insert(void* h, size_t pos, const char* text);
void  pt_erase(void* h, size_t start, size_t end);
int   pt_undo(void* h);            // 1 = ok, 0 = empty stack
int   pt_redo(void* h);
size_t pt_length(void* h);
char* pt_get_text(void* h);        // caller frees with pt_free_string
void  pt_free_string(char* s);
```
Strings are UTF-8; positions are byte offsets into the logical document.

### 5.2 Python sidecar (HTTP, JSON)
```
POST /lint            { "text": "..." }
        → { "errors": [...], "warnings": [...], "info": [...] }

POST /dataset/build   { "raw_dir": "dataset/raw" }
        → { "scripts": N, "accepted": N, "rejected": N,
             "train": N, "val": N }

POST /train           { "config": "AI/training/config.yaml" }
        → { "job": "train_<ts>" }

POST /generate        { "script": "script.txt", "reference": "reference_audio.wav" }
        → { "job": "gen_<ts>" }

GET  /status/:id      → { "state": "running|done|error",
                          "progress": 0.0–1.0,
                          "log_tail": [...],
                          "result": { ... } }   # e.g. output.mp3 duration
```

### 5.3 Tauri commands (Rust ↔ JS)
```ts
invoke("pt_insert", { pos, text })
invoke("pt_erase", { start, end })
invoke("pt_undo")
invoke("pt_redo")
invoke("pt_get_text") → string
invoke("spawn_sidecar") / invoke("kill_sidecar")
```

---

## 6. Data Pipeline

```
dataset/raw/*.txt
      │
      ▼
┌─────────────┐   fail ──► dataset/rejected/<name>.problems.txt
│   LINTER    │            (quarantine with reasons, never trained)
│  hard gate  │
└──────┬──────┘
       │ pass
       ▼
┌─────────────┐
│  SPLITTER   │  long scripts → continuation samples (≤ ~5.5k chars,
└──────┬──────┘  fits 2048-token training window)
       ▼
┌─────────────┐
│  SFT PAIRING│  system (format contract) + user (theme) + assistant (script)
└──────┬──────┘
       ▼
dataset/train.jsonl  (90%)
dataset/val.jsonl    (10%)
```

**Sourcing target:** 150–300 diverse scripts. Variety dimensions: induction style, theme, length, tag density. Persona-assisted generation is the primary collector; the linter is the quality gate.

---

## 7. Training Design — QLoRA on 6 GB

**Base model:** `Qwen/Qwen2.5-3B-Instruct` — largest practical instruct model for a 6 GB laptop GPU.

| Setting | Value | Reason |
|---|---|---|
| Quantization | 4-bit NF4, double quant | ~1.8 GB weights |
| LoRA rank / alpha | r=16, α=32 | capacity vs. VRAM balance |
| LoRA targets | q,k,v,o + gate,up,down | full-attention + MLP adaptation |
| Optimizer | paged_adamw_8bit | 8-bit states fit the 6 GB budget |
| Seq length | 2048 | matches section-split samples |
| Batch / grad accum | 1 × 16 | micro-batch 1, effective 16 |
| Gradient checkpointing | on | trades compute for activation VRAM |
| Loss | completion-only (assistant turns) | never train on the instruction |

**Estimated VRAM:** ~4.5–5.5 GB peak — fits with headroom on the RTX 3050 6 GB.
**Dataset gate:** training refuses to start with fewer than 30 accepted scripts (configurable) to avoid memorizing a single session.

---

## 8. Inference Design

### 8.1 Script writer (`AI/model/inference.py`)
- Loads base Qwen2.5-3B-Instruct + LoRA adapter (or merged weights).
- System prompt = the frozen format contract (same text as the dataset builder's `SYSTEM_PROMPT` — single source of truth).
- Sampling: temperature 0.8, top-p 0.9, repetition penalty 1.05; stop on closing markers.
- Output passes the linter before it is shown; violations are flagged inline in the UI, not silently accepted.

### 8.2 Session renderer (`generate.py`)
Per-paragraph pipeline: load reference clip → generate → trim silence → LUFS-normalize (−23) → stitch with 0.35 s gaps → peak-guard → WAV → ffmpeg MP3 (24 kHz mono, ~70 kbps).
**Budget:** ~27 s of audio per 600-char chunk; a 32k-char script ≈ 24 min.

---

## 9. Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 0 | TTS pipeline (chunked generation + MP3) | **done** |
| 0 | Format linter | **done** |
| 0 | Session GUI (Tkinter, interim) | **done** |
| 0 | Piece-table C++ core + ABI | core **done**, ABI planned |
| 1 | Directory scaffold (`AI/datapipeline`, `AI/model`, `AI/training`) | **done** |
| 1 | Dataset builder → JSONL under new layout | **done** (dual-homed, move pending) |
| 2 | Sidecar HTTP server (all four endpoints) | planned |
| 2 | QLoRA training script + config | planned |
| 2 | Dataset collection to 150+ scripts | in progress |
| 3 | Inference harness for the script writer | planned |
| 3 | Lit 3 frontend (all five panels) | planned |
| 4 | Tauri shell (requires rustup; FFI to C++ core, sidecar lifecycle) | planned |
| 4 | Node/koffi dev bridge (drop-in FFI substitute) | planned |
| 5 | Merge → GGUF export for portable script writer | planned |

---

## 10. Operational Limits & Failure Modes

- **VRAM:** training and TTS share the GPU — never run `/train` and `/generate` concurrently. The sidecar serializes GPU jobs.
- **Long scripts:** the writer has a 32k context; longer sessions must be split into continuation samples at dataset build time (already handled).
- **Reference audio:** < 10 s clips are rejected at the UI level with a duration probe.
- **Linter as gate:** no path from `dataset/raw/` to a dataset or to audio skips the linter.
- **Sidecar crash:** Tauri restarts it on demand; logs stream to the UI for diagnosis.
- **Git hygiene:** story content, audio, logs, checkpoints, and `node_modules` are `.gitignore`-blocked; only tooling is versioned.