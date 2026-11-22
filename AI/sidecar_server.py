# -*- coding: utf-8 -*-
"""
Sammonic sidecar: a local HTTP service that fronts every heavy operation.

Endpoints (see architecture.md §5.2):
    GET  /health
    POST /lint            { "text": "..." }
    POST /dataset/build   { "raw_dir": "dataset/raw" }
    POST /train           { "config": "AI/training/config.yaml" }
    POST /generate        { "script": ..., "reference": ... }
    POST /job/<id>/cancel
    GET  /status/<id>
    GET  /jobs

Design decisions that this file exists to enforce:

1. **Subprocess isolation.** train/generate run as child processes, not
   in-process imports. torch has to be loaded by the child so the server keeps
   its memory flat, and a crashed or cancelled job cannot corrupt the server
   or hold a GPU context open.
2. **One GPU at a time.** Training and TTS both saturate the GPU; running them
   concurrently OOMs a 6 GB card. A lock serializes them, and a request that
   arrives while one is running gets a 409 rather than being queued silently.
3. **A machine-readable heartbeat.** Jobs print `@@PROGRESS {...}` /
   `@@RESULT {...}` lines; the server parses those out of the log stream and
   never has to guess at progress by scraping human output.
4. **Bind to loopback only.** This is a local tool with no authentication.

Stdlib only, on purpose: the server must start on a bare Python install even
when torch/peft are broken or missing.
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ------------------------------------------------------------------ paths

ROOT = Path(os.environ.get("SAMMONIC_ROOT") or str(Path(__file__).resolve().parent.parent))
AI_DIR = ROOT / "AI"
LOG_DIR = AI_DIR / "logs"          # job stdout/stderr, git-ignored
DATAPIPE_DIR = AI_DIR / "datapipeline"
TRAIN_SCRIPT = AI_DIR / "training" / "train_qlora.py"
GEN_SCRIPT = ROOT / "generate.py"

HOST = os.environ.get("SAMMONIC_HOST", "127.0.0.1")
PORT = int(os.environ.get("SAMMONIC_PORT", "8765"))
MAX_BODY = 10 * 1024 * 1024        # 10 MB: a script plus slack, no more
LOG_TAIL_LINES = 120               # what /status returns

sys.path.insert(0, str(ROOT))      # so `AI.datapipeline...` imports

# The linter is imported lazily-ish: a broken build_dataset must not stop the
# server from serving /train and /generate. It is the only dependency of the
# whole process, and it is pure stdlib, so the failure mode is a bad edit.
from AI.datapipeline.build_dataset import (  # noqa: E402
    ALLOWED_TAGS,
    EMOJI_RE,
    lint_paragraphs,
)
from AI.piecetable_client import PieceTableUnavailable, Sessions  # noqa: E402

SESSIONS = Sessions()

# ------------------------------------------------------------------ jobs

# Kinds that need the GPU. Only one may run at a time.
GPU_KINDS = {"train", "generate"}

class GPU:
    """Serializes GPU work across the process.

    acquire() is non-blocking on purpose: a second GPU request must fail
    immediately with a 409 that names the job holding the card, not sit in a
    queue the caller cannot see.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._owner = None
        self._owner_lock = threading.Lock()

    def acquire(self, job_id):
        if not self._lock.acquire(blocking=False):
            return False
        with self._owner_lock:
            self._owner = job_id
        return True

    def occupied_by(self):
        with self._owner_lock:
            return self._owner

    def release(self, job_id):
        with self._owner_lock:
            if self._owner == job_id:
                self._owner = None
        self._lock.release()

gpu = GPU()


class Job:
    """One child process plus its parsed progress."""

    def __init__(self, kind, argv, cwd, meta=None):
        self.id = f"{kind}_{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.kind = kind
        self.argv = argv
        self.cwd = cwd
        self.meta = meta or {}
        self.state = "queued"
        self.progress = None       # last parsed @@PROGRESS payload
        self.result = None         # last parsed @@RESULT payload
        self.error = None
        self.returncode = None
        self.started_at = None
        self.ended_at = None
        self._log = deque(maxlen=LOG_TAIL_LINES)
        self._log_path = None
        self._log_file = None
        self._proc = None

    # -- log handling ---------------------------------------------------
    def _log_write(self, line):
        self._log.append(line)
        if self._log_file is not None:
            try:
                self._log_file.write(line + "\n")
                self._log_file.flush()
            except (OSError, ValueError):
                pass

    def start(self):
        """Spawn the child process and begin relaying its output."""
        self._log_path = LOG_DIR / f"{self.id}.log"
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            self._log_file = open(self._log_path, "w", encoding="utf-8", errors="replace")
        except OSError as exc:
            self.state, self.error = "error", f"cannot open log file: {exc}"
            return False

        self.started_at = time.time()
        self.state = "running"
        try:
            creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._proc = subprocess.Popen(
                self.argv,
                cwd=str(self.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation,
            )
        except OSError as exc:
            self.state = "error"
            self.error = f"failed to start process: {exc}"
            self._close_log()
            return False

        t = threading.Thread(target=self._drain, daemon=True)
        t.start()
        return True

    def _drain(self):
        """Read the child's output line by line, parsing heartbeats.

        Runs on its own thread so the request handler returns immediately and
        the child is never blocked on a full pipe buffer.
        """
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            self._log_write(line)
            if line.startswith("@@PROGRESS "):
                self.progress = _parse_json(line[len("@@PROGRESS "):], self.progress)
            elif line.startswith("@@RESULT "):
                self.result = _parse_json(line[len("@@RESULT "):], self.result)
        proc.wait()
        self.returncode = proc.returncode
        self.ended_at = time.time()
        if self.state == "cancelled":
            pass  # message already set by cancel()
        elif proc.returncode == 0:
            self.state = "done"
        else:
            self.state = "error"
            self.error = self.error or f"process exited with code {proc.returncode}"
        if self.kind in GPU_KINDS:
            gpu.release(self.id)
        self._close_log()

    def _close_log(self):
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

    # -- control --------------------------------------------------------
    def cancel(self):
        if self.state not in ("running", "queued"):
            return False
        self.state = "cancelled"
        self.error = "cancelled by request"
        self._log_write("[sidecar] cancel requested")
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                # Give it a moment to exit cleanly, then force.
                for _ in range(20):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.25)
                if proc.poll() is None:
                    proc.kill()
            except OSError:
                pass
        if self.kind in GPU_KINDS:
            gpu.release(self.id)
        return True

    # -- reporting ------------------------------------------------------
    def status(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "progress": self.progress,
            "returncode": self.returncode,
            "error": self.error,
            "elapsed_s": round(
                (self.ended_at or time.time()) - (self.started_at or time.time()), 2)
                if self.started_at else None,
            "log_tail": list(self._log),
            "result": self.result,
            "meta": self.meta,
        }


JOBS = {}
JOBS_LOCK = threading.Lock()

def _parse_json(text, fallback):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return fallback


def progress_fraction(job):
    """Best-effort 0..1 progress for a job, or None when unknown."""
    p = job.progress
    if not p:
        return None
    i, total = p.get("i"), p.get("total") or p.get("total_steps")
    if isinstance(i, (int, float)) and isinstance(total, (int, float)) and total > 0:
        return max(0.0, min(1.0, i / total))
    if isinstance(p.get("step"), (int, float)) and isinstance(p.get("total_steps"), (int, float)) \
            and p.get("total_steps", 0) > 0:
        return max(0.0, min(1.0, p["step"] / p["total_steps"]))
    return None


# ------------------------------------------------------------------ routes

def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "raw"


def route_lint(payload):
    """The hard format gate, run in-process (pure stdlib, no GPU, no torch).

    build_dataset.lint_paragraphs is the single source of truth for the rules;
    this route only classifies its output and adds statistics so the UI has
    something to show on a clean script. If the rules ever disagree, the
    dataset builder's answer wins: it is the one that decides what trains.
    """
    text = payload.get("text")
    if not isinstance(text, str):
        raise HttpError(400, "'text' must be a string")

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    errors, warnings, info = [], [], []

    for prob in lint_paragraphs(text):
        # These two are build_dataset's "allowed if intentional" cases.
        if prob.startswith(("para ", "no sound tags")):
            warnings.append(prob)
        else:
            errors.append(prob)

    # Rules the sidecar can answer without the dataset builder.
    if "\\" in text:
        errors.append("backslash present")
    if "*" in text and "((" in text:
        errors.append("double parentheses present (stage directions)")

    tags = sorted({t for t in re.findall(r"\[[a-z ]+\]", text) if t in ALLOWED_TAGS})
    info.append(f"paragraphs: {len(paras)}")
    info.append(f"chars: {len(text)}")
    if paras:
        sizes = [len(p) for p in paras]
        info.append(f"paragraph size range: {min(sizes)}-{max(sizes)}")
    info.append("tags: " + (", ".join(tags) if tags else "none"))

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }


def _start_job(kind, argv, cwd, meta, needs_gpu):
    job = Job(kind, argv, cwd, meta)
    if needs_gpu:
        if not gpu.acquire(job.id):
            owner = gpu.occupied_by()
            raise Busy(f"GPU busy with {owner}", owner)
    with JOBS_LOCK:
        JOBS[job.id] = job
    if not job.start():
        if needs_gpu:
            gpu.release(job.id)
        raise RuntimeError(job.error or f"{kind} job failed to start")
    return job


def route_dataset_build(payload):
    raw = (payload or {}).get("raw_dir") or str(ROOT / "dataset" / "raw")
    raw_path = Path(raw)
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    argv = [
        sys.executable, str(DATAPIPE_DIR / "build_dataset.py"),
        "--root", str(ROOT), "--raw", str(raw_path),
    ]
    job = _start_job("dataset", argv, ROOT, {"raw_dir": str(raw_path)}, needs_gpu=False)
    return {"job": job.id, "raw_dir": str(raw_path)}


def route_train(payload):
    cfg = (payload or {}).get("config") or str(AI_DIR / "training" / "config.yaml")
    cfg_path = Path(cfg)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not TRAIN_SCRIPT.exists():
        raise RuntimeError(f"training script not found: {TRAIN_SCRIPT}")
    argv = [sys.executable, str(TRAIN_SCRIPT), "--config", str(cfg_path)]
    resume = (payload or {}).get("resume")
    if resume:
        argv += ["--resume", str(resume)]
    job = _start_job("train", argv, ROOT, {"config": str(cfg_path)}, needs_gpu=True)
    return {"job": job.id, "config": str(cfg_path)}


def route_generate(payload):
    script = (payload or {}).get("script") or str(ROOT / "script.txt")
    reference = (payload or {}).get("reference") or str(ROOT / "reference_audio.wav")
    script_path, ref_path = Path(script), Path(reference)
    if not script_path.is_absolute():
        script_path = ROOT / script_path
    if not ref_path.is_absolute():
        ref_path = ROOT / ref_path

    # Fail before spawning: a missing input is a caller error, not a job error.
    if not script_path.exists():
        raise HttpError(400, f"script not found: {script_path}")
    if not ref_path.exists():
        raise HttpError(400, f"reference audio not found: {ref_path}")
    # The reference must be a usable voice clone, not a 200 ms stump.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(ref_path)],
        capture_output=True, text=True)
    try:
        ref_secs = float(probe.stdout.strip())
        if ref_secs < 3.0:
            raise HttpError(400, f"reference audio too short for cloning: {ref_secs:.1f}s")
    except ValueError as exc:
        raise HttpError(400, f"reference audio duration unreadable: "
                             f"{probe.stderr.strip()}") from exc

    argv = [
        sys.executable, str(GEN_SCRIPT),
        "--script", str(script_path),
        "--reference", str(ref_path),
    ]
    job = _start_job("generate", argv, ROOT,
                     {"script": str(script_path), "reference": str(ref_path)}, needs_gpu=True)
    return {"job": job.id, "script": str(script_path), "reference": str(ref_path)}


# ---------------------------------------------------------------- editing
# The editing core. These routes are what make the C++ piece table reachable
# from the UI before the Rust toolchain exists: no koffi, no Node addon, just
# ctypes loading the DLL built by app/cpp.

def _session(payload):
    sid = payload.get("session")
    if not isinstance(sid, str) or not sid:
        raise HttpError(400, "'session' is required")
    sess = SESSIONS.get(sid)
    if sess is None:
        raise HttpError(404, f"no such edit session: {sid}")
    if sess._closed:
        raise HttpError(410, f"session {sid} is closed")
    return sess


def _int(value, name, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool):
        raise HttpError(400, f"'{name}' must be an integer")
    if value < minimum:
        raise HttpError(400, f"'{name}' must be >= {minimum}")
    return value


def route_edit_open(payload):
    text = payload.get("text", "")
    if not isinstance(text, str):
        raise HttpError(400, "'text' must be a string")
    try:
        sess = SESSIONS.open(text)
    except PieceTableUnavailable as exc:
        raise HttpError(503, str(exc)) from exc
    return {"session": sess.id, "length": sess.length(), "pieces": sess.pieces()}


def route_edit_insert(payload):
    sess = _session(payload)
    text = payload.get("text")
    if not isinstance(text, str) or text == "":
        raise HttpError(400, "'text' must be a non-empty string")
    pos = _int(payload.get("pos", 0), "pos")
    with sess.lock:
        sess.insert(pos, text)          # clamps out-of-range positions
        return {"length": sess.length(), "pieces": sess.pieces(),
                "consistent": sess.consistent()}


def route_edit_erase(payload):
    sess = _session(payload)
    start = _int(payload.get("start", 0), "start")
    end = _int(payload.get("end", 0), "end")
    if start >= end:
        raise HttpError(400, "'end' must be greater than 'start' (erasing nothing is a no-op)")
    with sess.lock:
        sess.erase(start, end)
        return {"length": sess.length(), "pieces": sess.pieces(),
                "consistent": sess.consistent()}


def _undo_like(sess, fn, key):
    with sess.lock:
        did, text = fn()
        return {key: did, "length": sess.length(), "text": text}


def route_edit_undo(payload):
    sess = _session(payload)
    return _undo_like(sess, sess.undo, "undone")


def route_edit_redo(payload):
    sess = _session(payload)
    return _undo_like(sess, sess.redo, "redone")


def route_edit_text(payload):
    sess = _session(payload)
    with sess.lock:
        return {"text": sess.text(), "length": sess.length(),
                "pieces": sess.pieces(), "consistent": sess.consistent()}


def route_edit_close(payload):
    sid = payload.get("session")
    if not isinstance(sid, str) or not sid:
        raise HttpError(400, "'session' is required")
    return {"closed": SESSIONS.close(sid)}


def route_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise KeyError(job_id)
    body = job.status()
    body["progress_fraction"] = progress_fraction(job)
    return body


def route_cancel(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise KeyError(job_id)
    if not job.cancel():
        return {"id": job_id, "cancelled": False, "state": job.state,
                "detail": f"job already {job.state}"}
    return {"id": job_id, "cancelled": True, "state": job.state}


def route_jobs():
    with JOBS_LOCK:
        jobs = list(JOBS.values())
    return {
        "jobs": [{
            "id": j.id, "kind": j.kind, "state": j.state,
            "progress_fraction": progress_fraction(j), "error": j.error,
        } for j in jobs],
        "gpu_busy_with": gpu.occupied_by(),
    }


class HttpError(Exception):
    def __init__(self, status, message, extra=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra or {}


class Busy(HttpError):
    def __init__(self, message, owner):
        super().__init__(409, message, {"running_job": owner})


# ---------------------------------------------------------------- handler

class Handler(BaseHTTPRequestHandler):
    server_version = "SammonicSidecar/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing -------------------------------------------------------
    def log_message(self, fmt, *args):
        # The sidecar's own access log is noise next to job logs; keep stdout
        # for job output only.
        pass

    def _send(self, status, body):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise HttpError(400, "Content-Length is not a number")
        if length > MAX_BODY:
            raise HttpError(413, f"body too large (limit {MAX_BODY} bytes)")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        if not raw.strip():
            return {}
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise HttpError(400, f"body is not valid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise HttpError(400, "body must be a JSON object")
        return body

    # -- verbs ----------------------------------------------------------
    def _dispatch(self, verb):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if verb == "GET":
                if path == "/health":
                    return self._send(200, {"status": "ok",
                                            "gpu_busy_with": gpu.occupied_by()})
                if path == "/jobs":
                    return self._send(200, route_jobs())
                m = re.fullmatch(r"/status/([A-Za-z0-9_.\-]+)", path)
                if m:
                    return self._send(200, route_status(m.group(1)))
                return self._send(404, {"error": f"unknown path: {path}"})
            if verb == "POST":
                body = self._read_body()
                # Explicit routing, one branch per endpoint, so a typo in a
                # table literal can never silently mis-route a job.
                if path == "/lint":
                    return self._send(200, route_lint(body))
                if path == "/dataset/build":
                    job = route_dataset_build(body)
                    return self._send(202, job)
                if path == "/train":
                    job = route_train(body)
                    return self._send(202, job)
                if path == "/generate":
                    job = route_generate(body)
                    return self._send(202, job)
                m = re.fullmatch(r"/job/([A-Za-z0-9_.\-]+)/cancel", path)
                if m:
                    return self._send(200, route_cancel(m.group(1)))
                if path == "/edit/open":
                    return self._send(200, route_edit_open(body))
                if path == "/edit/insert":
                    return self._send(200, route_edit_insert(body))
                if path == "/edit/erase":
                    return self._send(200, route_edit_erase(body))
                if path == "/edit/undo":
                    return self._send(200, route_edit_undo(body))
                if path == "/edit/redo":
                    return self._send(200, route_edit_redo(body))
                if path == "/edit/text":
                    return self._send(200, route_edit_text(body))
                if path == "/edit/close":
                    return self._send(200, route_edit_close(body))
                return self._send(404, {"error": f"unknown path: {path}"})
            return self._send(405, {"error": f"{verb} not allowed on {path}"})
        except HttpError as exc:
            extra = {"error": exc.message}
            extra.update(exc.extra)
            return self._send(exc.status, extra)
        except KeyError as exc:
            return self._send(404, {"error": f"unknown job: {exc.args[0] if exc.args else '?'}"})
        except Exception as exc:  # never leak a traceback to the client, never kill the server
            return self._send(500, {"error": f"internal error: {exc}",
                                    "type": type(exc).__name__})

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"Sammonic sidecar listening on http://{HOST}:{PORT}", flush=True)
    print(f"project root: {ROOT}", flush=True)
    print(f"job logs: {LOG_DIR}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("\nsidecar stopped", flush=True)


if __name__ == "__main__":
    main()