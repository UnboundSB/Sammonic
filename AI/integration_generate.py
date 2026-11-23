# -*- coding: utf-8 -*-
"""
End-to-end integration test: a real TTS render driven through the sidecar.

    python AI/sidecar_server.py &            # in one shell
    python AI/integration_generate.py        # in another

This is the one seam unit tests cannot cover: the job spinning up as a child
process, heartbeats flowing through the server's log parser, the GPU lock
being taken and released, and the final @@RESULT payload becoming a real file
on disk.

It writes to a scratch path, never the project's output.mp3, so running it
does not destroy a session the user has already approved.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("SAMMONIC_ROOT") or str(Path(__file__).resolve().parent.parent))
PORT = os.environ.get("SAMMONIC_PORT", "8765")
BASE = f"http://127.0.0.1:{PORT}"

SCRATCH = ROOT / "AI" / "logs" / "integration_scratch.wav"
SCRATCH_MP3 = ROOT / "AI" / "logs" / "integration_scratch.mp3"
TIMEOUT_S = 45 * 60


def post(path, body):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or b"{}")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.status, json.loads(r.read().decode())


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def main():
    stamp = time.strftime("%H:%M:%S")
    print(f"== integration: /generate -> real MP3 ==  ({stamp})")

    status, body = post("/generate", {
        "script": "script.txt",
        "reference": "reference_audio.wav",
        "out_wav": "AI/logs/integration_scratch.wav",
        "out_mp3": "AI/logs/integration_scratch.mp3",
    })
    if status != 202:
        print(f"FAIL: /generate returned {status}: {body}")
        return 1
    job = body["job"]
    print(f"job started: {job}")

    seen, last = 0, None
    started = time.time()
    while time.time() - started < TIMEOUT_S:
        _, st = get(f"/status/{job}")
        last = st
        if st["state"] in ("done", "error", "cancelled"):
            break
        p = st.get("progress") or {}
        frac = st.get("progress_fraction")
        if frac is not None and frac > 0:
            seen = max(seen, frac)
            print(f"  chunk {p.get('i')}/{p.get('total')}  "
                  f"{frac * 100:5.1f}%  {p.get('secs')}s this chunk", flush=True)
        else:
            print(f"  {st['state']}...", flush=True)
        time.sleep(10)

    if last is None or last["state"] != "done":
        print(f"FAIL: job ended in state {last and last['state']}, "
              f"rc={last and last.get('returncode')}, error={last and last.get('error')}")
        tail = "\n".join((last or {}).get("log_tail", [])[-15:])
        print("log tail:\n" + tail)
        return 1

    print(f"\njob done in {last['elapsed_s']:.0f}s (wall clock)")
    result = last.get("result") or {}
    print(f"@@RESULT parsed by the sidecar: {json.dumps(result)}")

    problems = []
    if result.get("duration") is None:
        problems.append("no duration in the parsed result")

    if not SCRATCH_MP3.exists():
        problems.append(f"output file missing: {SCRATCH_MP3}")
    else:
        actual = probe(SCRATCH_MP3)
        print(f"ffprobe duration: {actual}s")
        if actual is None:
            problems.append("ffprobe could not read the output duration")
        elif result.get("duration") and abs(actual - result["duration"]) > 5.0:
            problems.append(f"result duration {result['duration']} disagrees with "
                           f"ffprobe {actual} by more than 5s")

    if not last.get("progress"):
        problems.append("no @@PROGRESS payload was ever parsed")

    _, jobs = get("/jobs")
    if jobs.get("gpu_busy_with"):
        problems.append(f"GPU lock was never released: {jobs['gpu_busy_with']}")

    # The real session must be untouched by this test.
    if (ROOT / "output.mp3").exists():
        size = (ROOT / "output.mp3").stat().st_size
        print(f"project output.mp3 still present ({size} bytes) - not clobbered")
    else:
        print("note: project output.mp3 does not exist (nothing to protect)")

    print("\n== summary ==")
    if problems:
        for p in problems:
            print("  FAIL:", p)
        return 1
    print("generate works end to end through the sidecar")
    return 0


if __name__ == "__main__":
    sys.exit(main())