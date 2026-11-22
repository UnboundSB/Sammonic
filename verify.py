# -*- coding: utf-8 -*-
"""
One command to prove the system still works.

    python verify.py

Runs, in order:
  1. Python syntax check on every tracked .py file
  2. The C++ engine and ABI test binaries (building them first if needed)
  3. The full sidecar self-test against a live server on a scratch port

Deliberately does NOT run: a TTS render (~20 min of GPU), a QLoRA run
(needs libraries that may not be installed), or anything that writes to
`output.wav` / `output.mp3`.

Exit code 0 means every check above passed; any failure prints what failed
and exits non-zero, because a toolkit you cannot verify is a liability.
"""

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("SAMMONIC_ROOT") or str(Path(__file__).resolve().parent))
RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n      {detail}" if detail else ""))
    return ok


def run(cmd, cwd=None, timeout=600, env=None):
    return subprocess.run(cmd, cwd=str(cwd or ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, env=env)


# ---------------------------------------------------------------- python
def check_python():
    files = sorted(p for p in ROOT.rglob("*.py")
                   if "node_modules" not in str(p) and "__pycache__" not in str(p)
                   and not str(p.relative_to(ROOT)).startswith("_"))
    proc = run([sys.executable, "-m", "py_compile", *[str(p) for p in files]])
    return record(f"python syntax ({len(files)} files)", proc.returncode == 0,
                  proc.stderr.strip()[:400])


# ---------------------------------------------------------------- C++
def check_cpp():
    build = ROOT / "app" / "cpp" / "build"
    binaries = {
        "C++ engine tests": build / "piecetable_test.exe",
        "C++ ABI tests": build / "c_api_test.exe",
    }
    need_build = not all(p.exists() for p in binaries.values())
    if need_build:
        print("... building the C++ core (first run)")
        cfg = run(["cmake", "-S", str(ROOT / "app" / "cpp"), "-B", str(build),
                   "-G", "MinGW Makefiles"], timeout=300)
        if cfg.returncode != 0:
            return record("C++ build", False, cfg.stderr.strip()[:400])
        bld = run(["cmake", "--build", str(build)], timeout=600)
        if bld.returncode != 0:
            return record("C++ build", False, bld.stderr.strip()[:400])

    ok = True
    for label, binary in binaries.items():
        proc = run([str(binary)])
        tail = (proc.stdout or "").strip().splitlines()
        ok &= record(label, proc.returncode == 0,
                     tail[-1] if tail else proc.stderr.strip()[:200])
    return ok


# ---------------------------------------------------------------- sidecar
def check_sidecar():
    port = os.environ.get("SAMMONIC_VERIFY_PORT", "8799")
    # The self-test reads the port from the environment, so it must inherit
    # the same override the server was started with — otherwise the test
    # looks for a server on the default port and reports a confusing failure.
    env = dict(os.environ, SAMMONIC_PORT=port)
    proc = subprocess.Popen([sys.executable, str(ROOT / "AI" / "sidecar_server.py")],
                            cwd=str(ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    try:
        # Wait for the server to accept connections before testing it.
        base = f"http://127.0.0.1:{port}/health"
        for _ in range(60):
            try:
                with urllib.request.urlopen(base, timeout=2) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        else:
            return record("sidecar self-test", False, "server never came up")

        test = run([sys.executable, str(ROOT / "AI" / "selftest_sidecar.py")],
                   env=env, timeout=300)
        return record("sidecar self-test", test.returncode == 0,
                      (test.stdout or "").strip()[-500:])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    print("== Sammonic verification ==\n")
    check_python()
    check_cpp()
    check_sidecar()

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n== summary ==")
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("failed:")
        for n in failed:
            print("  -", n)
        sys.exit(1)
    print("everything passes")