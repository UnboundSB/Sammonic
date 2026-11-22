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
│  │  Lit 3 web components │   │  Tauri 2 shell (Rust)            │   │
│  │ - ScriptEditor       │   │ - window, menus, file dialogs   │   │
│  │ - LintPanel          │   │ - sidecar lifecycle             │   │
│  │ - DatasetPanel       │   │                                │   │
│  │ - TrainPanel         │   │                                │   │
│  │ - GeneratePanel      │   │                                │   │
│  └──────────┬────────────┘   └───────────────┬────────────────┘   │
│             │  fetch() / custom events       │  tauri::command      │
└─────────────┼───────────────────────────────┼──────────────────────┘
               │                               │
               │  ONE origin: the sidecar       │  Rust not installed:
               │                               │  the shell is deferred
               ▼                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        PYTHON SIDECAR                                │
│  AI/sidecar_server.py  (stdlib http.server, no dependencies)         │
│  binds 127.0.0.1:8765 — loopback only, no auth needed locally        │
│                                                                      │
│  ┌────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ ┌───────────┐      │
│  │ /lint  │ │ /dataset │ │ /train │ │/generate │ │ /edit/*   │      │
│  │ gate   │ │  build   │ │ QLoRA  │ │   TTS    │ │  editor   │      │
│  └───┬────┘ └────┬─────┘ └───┬────┘ └────┬─────┘ └─────┬─────┘      │
└──────┬───────────┬───────────┬───────────┬─────────────┬────────────┘
       ▼           ▼           ▼           ▼             ▼
   in-process  build_dataset   train_qlora  generate.py   ctypes
   (stdlib)    .py             .py                        ▼
                                                        ┌────────────┐
                                                        │ libpiece-  │
                                                        │ table.dll  │
                                                        │  C ABI     │
                                                        └────────────┘
       │           │           │           │
       ▼           ▼           ▼           ▼
  lint report  dataset/*.jsonl  AI/model/   output.mp3
               dataset/raw/    adapters
               dataset/rejected/
```

**One origin, by design.** An earlier plan had a Node + koffi bridge on the
same port standing in for the Rust FFI layer. That plan is gone: the C++
core is reachable from Python through stdlib `ctypes`, so the sidecar serves
`/edit/*` itself and the UI has exactly one HTTP origin. It also means the
editing core is exercised through the same code path in dev and in
production, with no second implementation to keep in sync.

Long jobs (`train`, `generate`, `dataset`) run as child processes so the
server never imports torch, and so a crashed or cancelled job cannot take
the server with it.

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

**Dev mode:** `npm run dev` (Vite) serves the Lit app. It talks to the sidecar on one origin
(`VITE_SAMMONIC_API`, default `http://127.0.0.1:8765`); the sidecar's `/edit/*` routes are
backed by the C++ core through `ctypes`, so the UI runs before the Rust toolchain exists and
no bridge process is needed. Editing traffic is serialized client-side and again by a
per-session lock in the sidecar, because an out-of-order edit corrupts byte offsets.

### 3.2 C++ Editing Core

- **File:** `app/cpp/piece_table.{h,cpp}` — piece table with append-only `orig`/`add` buffers, coalescing on the ADD hot path, undo/redo via forward-edit op stacks.
- **Why C++:** edits on a multi-MB script stay flat in RAM (never materialize the full string except for `slice`), and undo history costs nothing extra.
- **C ABI** (`c_api.cpp`): opaque handle + `pt_create`, `pt_destroy`, `pt_insert`, `pt_erase`, `pt_undo`, `pt_redo`, `pt_length`, `pt_get_text`, `pt_free_string`, plus `pt_piece_count` and `pt_check_invariant` as diagnostics. Every entry point null-checks its handle (a null handle must be inert, never fatal) and one process-wide recursive mutex serializes all of them.
- **MinGW builds link `-static-libgcc -static-libstdc++ -static`**, otherwise the DLL carries hidden dependencies on the MSYS2 runtimes and fails to load on a machine without them on `PATH`.
- **Build:** CMake. `cmake -S app/cpp -B app/cpp/build && cmake --build app/cpp/build` produces `libpiecetable.dll` plus two test binaries.
- **Tests:** `test_piecetable.cpp` (engine: 233 checks) and `test_c_api.cpp` (ABI, incl. null handles and 64 concurrent handles). Both currently pass with zero failures.

### 3.3 Python Service Layer

Single sidecar process (`AI/sidecar_server.py`, stdlib only). Jobs are child processes, not
in-process imports, so torch is never loaded by the server and a crashed job cannot take the
server down.

| Endpoint | Job | Script |
|---|---|---|
| `POST /lint` | Format gate — chunk sizes, tag legality, emoji, SSML, stage directions | linter inside `build_dataset.py` (in-process; pure stdlib) |
| `POST /dataset/build` | `dataset/raw/*.txt` → validated `train.jsonl`/`val.jsonl`; violators to `dataset/rejected/` with reasons | `AI/datapipeline/build_dataset.py` |
| `POST /train` | QLoRA fine-tune run (async; streams progress) | `AI/training/train_qlora.py` |
| `POST /generate` | Chatterbox TTS session render (async; streams chunk progress) | `generate.py` |
| `GET /status/:id` | Job progress polling | — |
| `POST /job/:id/cancel` | Terminates the child process (SIGTERM, then SIGKILL) | — |
| `GET /jobs` | Every job, plus which one holds the GPU | — |
| `POST /edit/open` … `/edit/close` | Document editing against the C++ piece table | `AI/piecetable_client.py` (ctypes → `libpiecetable.dll`) |

Long jobs return `202` with a job id and the frontend polls `/status`. `train` and `generate`
both need the GPU, and only one may run at a time on a 6 GB card: the sidecar serializes them
and answers `409` with the id of the job already holding it rather than queueing silently.

Every job prints `@@PROGRESS {...}` / `@@RESULT {...}` lines; the sidecar parses those instead
of scraping human-readable output, so progress stays machine-checkable.

---

## 4. Directory Map

```
D:\hypno maker\
├── architecture.md          this document
├── README.md                repo overview
├── generate.py              Chatterbox TTS pipeline
├── .gitignore               blocks: content, audio, logs, venv, checkpoints
│
├── app/
│   ├── cpp/                 piece-table editing core
│   │   ├── piece_table.h        engine
│   │   ├── piece_table.cpp      implementation
│   │   ├── c_api.cpp            extern "C" bridge
│   │   ├── CMakeLists.txt       build
│   │   ├── test_piecetable.cpp  engine tests   (233 checks, passing)
│   │   ├── test_c_api.cpp       ABI tests      (46 checks, passing)
│   │   └── build/               git-ignored output
│   └── web/                 Lit 3 + Vite frontend
│       ├── package.json / vite.config.ts / tsconfig.json
│       └── src/
│           ├── api.ts            typed client for every route
│           ├── state.ts          small subscribe/emit store
│           ├── hypno-app.ts      shell + panels
│           ├── script-editor.ts  textarea bound to the piece table
│           ├── lint-panel.ts     debounced lint results
│           ├── dataset-panel.ts  dataset build + counts
│           ├── train-panel.ts    training progress + logs
│           └── generate-panel.ts TTS render progress + duration
│
├── AI/
│   ├── sidecar_server.py    HTTP service: /lint, /dataset, /train,
│   │                        /generate, /edit, /status, /jobs
│   ├── selftest_sidecar.py  end-to-end self-test (run against a live server)
│   ├── piecetable_client.py ctypes bridge to libpiecetable.dll
│   ├── datapipeline/
│   │   ├── build_dataset.py     linter + SFT JSONL builder
│   │   └── (SCHEMA.md lives in dataset/)
│   ├── model/
│   │   ├── inference.py     script-writer adapter loading + sampling
│   │   ├── adapters/        LoRA checkpoints (git-ignored)
│   │   └── merged/          merged full weights (git-ignored)
│   └── training/
│       ├── config.yaml      hyperparameters, paths, dataset gate
│       ├── train_qlora.py   4-bit QLoRA training loop
│       └── events.jsonl     live telemetry (git-ignored)
│
├── hypno_maker.py          Tkinter GUI (interim UI)
├── writer.py               line-by-line drafting loop
├── strip_emoji.py          content scrubber
│
└── dataset/
    ├── raw/                source scripts (git-ignored — content)
    ├── train.jsonl / val.jsonl   derived SFT data (git-ignored — content)
    ├── rejected/           linter failures + reasons (git-ignored)
    └── SCHEMA.md           dataset contract
```

Everything that touches the GPU runs as a child process of the sidecar; the
sidecar itself imports only `http.server`, `json`, and `ctypes`, and starts
even when torch, peft, and trl are broken or missing.

---

## 5. IPC Contracts

### 5.1 C++ core (C ABI)
```c
void*        pt_create(const char* initial_text);   // null handle => inert
void         pt_destroy(void* handle);              // destroying null is a no-op
void         pt_insert(void* h, size_t pos, const char* text);
void         pt_erase(void* h, size_t start, size_t end);
int          pt_undo(void* h);      // 1 = applied, 0 = stack empty
int          pt_redo(void* h);
size_t       pt_length(void* h);
char*        pt_get_text(void* h);  // caller frees with pt_free_string
void         pt_free_string(char* s);
size_t       pt_piece_count(void* h);     // diagnostics: coalescing health
int          pt_check_invariant(void* h); // 1 = piece list matches length
```

All strings are UTF-8 and all offsets are **byte** offsets into the logical
document. Every entry point null-checks its handle — a null handle is inert,
never fatal — and one process-wide `std::recursive_mutex` serializes all of
them, so a handle can be touched from the UI thread and a job worker at the
same time. Every `char*` returned must be released with `pt_free_string`.

### 5.2 Python sidecar (HTTP, JSON)
Base URL `http://127.0.0.1:8765` (`SAMMONIC_HOST` / `SAMMONIC_PORT`).

```
POST /lint            { "text": "..." }
        → { "ok": bool, "errors": [...], "warnings": [...], "info": [...] }

POST /dataset/build   { "raw_dir": "dataset/raw" }        # optional
        → 202 { "job": "dataset_<ts>_<rand>" }

POST /train           { "config": "AI/training/config.yaml", "resume": null }
        → 202 { "job": "train_<ts>_<rand>" }   | 409 if the GPU is busy

POST /generate        { "script": "...", "reference": "..." }
        → 202 { "job": "gen_<ts>_<rand>" }     | 409 if the GPU is busy
        → 400 if the script or reference is missing/unusable

GET  /status/:id      → { "id", "kind",
                          "state": "queued|running|done|error|cancelled",
                          "progress": <last @@PROGRESS payload>,
                          "progress_fraction": 0.0–1.0,
                          "returncode", "error", "elapsed_s",
                          "log_tail": [...], "result": {...} }

POST /job/:id/cancel → { "id", "cancelled": true, "state" } | 404
GET  /jobs           → { "jobs": [...], "gpu_busy_with": <job id|null> }

POST /edit/open       { "text": "..." }        → { "session", "length", "pieces" }
POST /edit/insert     { "session", "pos", "text" }   # pos is a byte offset
POST /edit/erase      { "session", "start", "end" }  # end > start required
POST /edit/undo       { "session" }  → { "undone": bool, "length", "text": full doc }
POST /edit/redo       { "session" }  → { "redone": bool, "length", "text": full doc }
POST /edit/text       { "session" }  → { "text", "length", "pieces", "consistent" }
POST /edit/close      { "session" }
```

Errors are JSON with an `error` key and a 4xx/5xx status: 400 malformed body
or invalid arguments, 404 unknown job or route, 405 wrong verb, 409 GPU busy,
410 closed session, 413 body over the limit, 500 unexpected server fault.

`/edit/open` returns 503 when the C++ library has not been built, with the
exact CMake commands to build it.

### 5.3 Tauri commands (deferred — Rust is not installed)
```ts
invoke("pt_insert", { pos, text })   // or, in the shipped boot path, POST /edit/insert
invoke("pt_undo") / invoke("pt_redo")
invoke("spawn_sidecar") / invoke("kill_sidecar")
```

The Rust shell is **deferred**: `rustc` and `cargo` are not present on the
target machine, and neither is MSVC. Until they are, the Lit UI talks to the
sidecar directly, and the sidecar reaches the C++ core through `ctypes`. The
Tauri layer adds only window management and native dialogs, so deferring it
costs nothing functionally.

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

**Base model:** `Qwen/Qwen3-4B-Instruct-2507` — newer writing quality than the 2.5 generation
and still small enough for 6 GB QLoRA. `Qwen/Qwen2.5-3B-Instruct` remains the fallback if the
4B will not fit or its output proves worse on held-out samples. Selection is a single
`base_model` key in `AI/training/config.yaml`, not a code change.

| Setting | Value | Reason |
|---|---|---|
| Quantization | 4-bit NF4, double quant | ~1.8 GB weights |
| LoRA rank / alpha | r=16, α=32 | capacity vs. VRAM balance |
| LoRA targets | q,k,v,o + gate,up,down | full-attention + MLP adaptation |
| Optimizer | paged_adamw_8bit | 8-bit states fit the 6 GB budget |
| Seq length | 2048 | matches section-split samples (~5.5k chars, ~2k tokens) |
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
| 0 | TTS pipeline (chunked generation + MP3) | **done and verified** — 52/52 chunks, 23:14 |
| 0 | Format linter | **done** |
| 0 | Session GUI (Tkinter, interim) | **done** |
| 0 | Piece-table C++ core + C ABI | **done, tested** — 233 engine + 46 ABI checks passing |
| 1 | Directory scaffold (`AI/datapipeline`, `AI/model`, `AI/training`) | **done** |
| 1 | Dataset builder under the new layout | **done** — moved to `AI/datapipeline/`, root override added |
| 1 | ctypes bridge to the editor core | **done** — `AI/piecetable_client.py` |
| 2 | Sidecar HTTP server (all routes, incl. `/edit/*`) | **done and self-tested** |
| 2 | Lit 3 frontend (all five panels) | **done and type-checked** — `tsc --noEmit` clean, production bundle built |
| 2 | QLoRA training script + config | in progress |
| 2 | Dataset collection to 150+ scripts | **blocked** — 1 script available; gate at 30 |
| 3 | Inference harness for the script writer | in progress |
| 4 | Tauri shell | **blocked** — no Rust, no MSVC on the target machine |
| 5 | Merge → GGUF export for a portable script writer | blocked on training |

### What is verified versus merely written

Every claim in the table above was checked by running it, not by inspection:

- C++ engine and ABI: both test binaries run, zero failures, zero warnings.
- `libpiecetable.dll` loads through Python `ctypes` and round-trips
  insert/erase/undo/redo, including multi-byte UTF-8.
- The sidecar answers every route, including the failure paths (400 malformed
  JSON, 404 unknown job/route, 409 GPU busy, 410 closed session, 503 missing
  DLL). `AI/selftest_sidecar.py` runs the whole surface against a live server.
- `build_dataset.py` and `generate.py` both still work after being moved and
  refactored; `generate.py` was smoke-tested end to end (two chunks to MP3).
- The frontend compiles and builds with strict TypeScript.

Not verified, and why: a real QLoRA run (needs peft/trl/bitsandbytes and a GPU
session long enough to watch it), a full TTS run through the sidecar (takes
~20 minutes of GPU), and anything in a browser that has not been driven by an
automation tool.

---

## 10. Operational Limits & Failure Modes

- **VRAM:** training and TTS share the GPU — never run `/train` and `/generate` concurrently. The sidecar serializes GPU jobs.
- **Long scripts:** the writer has a 32k context; longer sessions must be split into continuation samples at dataset build time (already handled).
- **Reference audio:** < 10 s clips are rejected at the UI level with a duration probe.
- **Linter as gate:** no path from `dataset/raw/` to a dataset or to audio skips the linter.
- **Sidecar crash:** Tauri restarts it on demand; logs stream to the UI for diagnosis.
- **Git hygiene:** story content, audio, logs, checkpoints, and `node_modules` are `.gitignore`-blocked; only tooling is versioned.