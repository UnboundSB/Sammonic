# -*- coding: utf-8 -*-
"""Self-test for the sidecar. Run while the server is up:

    python AI/sidecar_server.py &          # in one shell
    python AI/selftest_sidecar.py          # in another

Exercises every route, including the failure paths that matter: bad JSON,
unknown job, GPU contention, and a job that must not start.
Not a unit-test substitute: it checks the wiring end to end over real HTTP.
"""

import json
import sys
import urllib.error
import urllib.request

BASE = f"http://{__import__('os').environ.get('SAMMONIC_HOST','127.0.0.1')}:" \
       f"{__import__('os').environ.get('SAMMONIC_PORT','8765')}"

GOOD = ("This is the first paragraph of a completely ordinary script, written "
        "long enough to satisfy the linter and to describe a calm induction. "
        "[sigh] Let the words settle.\n\n"
        "Here is the second paragraph, also long enough to pass, carrying the "
        "same feeling of slow deliberate descent into a comfortable focus that "
        "deepens with every measured breath. [chuckle] And a little more text.")

failed = []


def call(method, path, body=None, expect=200):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            status, payload = r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        status = e.code
        payload = json.loads(e.read().decode() or b"{}")
    ok = status == expect
    print(f"{'ok ' if ok else 'FAIL'} {method} {path} -> {status}"
          + ("" if ok else f" (want {expect})"))
    if not ok:
        failed.append(f"{method} {path} returned {status}, want {expect}")
    return status, payload


def check(label, cond):
    print(f"{'ok ' if cond else 'FAIL'} {label}")
    if not cond:
        failed.append(label)


print(f"== sidecar self-test against {BASE} ==\n")

st, body = call("GET", "/health")
check("health reports status ok", body.get("status") == "ok")

st, body = call("GET", "/jobs")
check("jobs returns a list", isinstance(body.get("jobs"), list))

# --- lint: accept, warn, and reject ---------------------------------
st, body = call("POST", "/lint", {"text": GOOD})
check("clean script lints ok", body["ok"] is True)
check("clean script has stats", any(i.startswith("paragraphs:") for i in body["info"]))

st, body = call("POST", "/lint", {"text": "too short. [bogus tag] <xml> emoji \U0001F600"})
check("broken script fails lint", body["ok"] is False)
check("broken script lists illegal tag", any("illegal tags" in e for e in body["errors"]))
check("broken script lists emoji", any("emoji" in e for e in body["errors"]))

call("POST", "/lint", {"text": 42}, expect=400)
call("POST", "/lint", expect=400)

# --- transport-level failures ---------------------------------------
req = urllib.request.Request(BASE + "/lint", data=b"{not json",
                             method="POST", headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=10)
    check("malformed JSON rejected", False)
except urllib.error.HTTPError as e:
    check("malformed JSON rejected", e.code == 400)

call("GET", "/status/does_not_exist", expect=404)
call("POST", "/no_such_route", {}, expect=404)
call("PUT", "/lint", {}, expect=405)

# --- dataset build (CPU job, safe to actually run) ------------------
# 202 Accepted: the job starts in the background, that is the contract.
st, body = call("POST", "/dataset/build", {}, expect=202)
job_id = body.get("job")
check("dataset build returned a job id", bool(job_id))

st, body = call("GET", f"/status/{job_id}")
check("job state present", body["state"] in ("queued", "running", "done", "error"))

import time                                  # noqa: E402
deadline = time.time() + 60
while time.time() < deadline:
    with urllib.request.urlopen(BASE + f"/status/{job_id}", timeout=15) as r:
        body = json.loads(r.read().decode())
    if body["state"] in ("done", "error", "cancelled"):
        break
    time.sleep(1.0)

if body["state"] == "done":
    check("dataset build produced a result payload", bool(body.get("result")))
    check("dataset build produced a progress payload", bool(body.get("progress")))
    print("    result:", json.dumps(body.get("result")))
elif body["state"] == "error":
    # The script currently lints clean on the single seed script; an error
    # here would be a real problem worth reporting verbatim.
    print("    job error:", body.get("error"))
    print("    log tail:\n", "\n".join(body.get("log_tail", [])[-12:]))
    check("dataset build succeeded", False)
else:
    check("dataset build completed", False)

# --- unknown job -----------------------------------------------------
call("POST", "/job/nope/cancel", {}, expect=404)
st, body = call("GET", "/jobs")
check("jobs lists the finished job", any(j["id"] == job_id for j in body["jobs"]))

# --- generate without the script must fail fast, not start a job ----
call("POST", "/generate", {"script": "definitely_not_here.txt"}, expect=400)
call("POST", "/generate", {"script": "script.txt", "reference": "definitely_not_here.wav"}, expect=400)

# --- train: job lifecycle, including a job that must refuse -------------
# The trainer exists now, so /train returns 202 and the child runs until it
# hits something real (missing peft, or the dataset gate) and exits non-zero.
# Verifying that path end to end matters more than expecting a 500: a trainer
# that fails loudly must produce a failed *job*, never a crashed server.
st, body = call("POST", "/train", {"config": "AI/training/config.yaml"}, expect=202)
train_job = body.get("job")
check("train returned a job id", bool(train_job))

tbody = {}
if train_job:
    deadline = time.time() + 120
    while time.time() < deadline:
        with urllib.request.urlopen(BASE + f"/status/{train_job}", timeout=15) as r:
            tbody = json.loads(r.read().decode())
        if tbody["state"] in ("done", "error", "cancelled"):
            break
        time.sleep(1.0)

    if tbody["state"] == "error":
        check("failed train job reports a non-zero return code",
              tbody.get("returncode") not in (0, None))
        refusal = next((l for l in tbody.get("log_tail", [])
                        if "TRAINING REFUSED" in l), None)
        check("failed train job explains itself on stderr",
              refusal is not None or "missing training dependency" in
              "\n".join(tbody.get("log_tail", [])))
        print("    train job ended rc=%s | %s" % (tbody.get("returncode"),
                                                  (refusal or "(no refusal line)")[:120]))
    elif tbody["state"] == "done":
        check("train job completed (deps installed, gate cleared)", True)
        print("    train job completed with result:", tbody.get("result"))
    else:
        check("train job reached a terminal state", False)

    # The GPU lock must be released when a job dies, or every later job is
    # answered 409 forever.
    st, jb = call("GET", "/jobs")
    check("GPU released after the job ended", not jb.get("gpu_busy_with"))

# --- editing core (ctypes -> piece table) ---------------------------
st, body = call("POST", "/edit/open", {"text": "hello world"})
check("edit session opened", bool(body.get("session")))
check("session reports byte length", body.get("length") == 11)
sid = body.get("session")

st, body = call("POST", "/edit/insert", {"session": sid, "pos": 5, "text": " THERE"})
check("insert returns new length", body.get("length") == 17)
check("insert keeps the invariant", body.get("consistent") is True)

st, body = call("POST", "/edit/text", {"session": sid})
check("text reflects the insert", body.get("text") == "hello THERE world")

st, body = call("POST", "/edit/undo", {"session": sid})
check("undo reports undone", body.get("undone") is True)
check("undo restores the text", body.get("text") == "hello world")

st, body = call("POST", "/edit/redo", {"session": sid})
check("redo reports redone", body.get("redone") is True)
check("redo reapplies the edit", body.get("text") == "hello THERE world")

call("POST", "/edit/erase", {"session": sid, "start": 5, "end": 5}, expect=400)
call("POST", "/edit/insert", {"session": sid, "pos": 0, "text": ""}, expect=400)
call("POST", "/edit/insert", {"session": "ed_does_not_exist", "pos": 0, "text": "x"}, expect=404)
call("POST", "/edit/text", {"session": sid}, expect=200)

st, body = call("POST", "/edit/open", {"text": "caf\u00e9 \u20ac \u00e9\u00e8\u00ea"})
check("utf-8 session opens", bool(body.get("session")))
u = body["session"]
st, body = call("POST", "/edit/text", {"session": u})
check("utf-8 round trips losslessly", body.get("text") == "caf\u00e9 \u20ac \u00e9\u00e8\u00ea")

print("\n== summary ==")
if failed:
    print(f"{len(failed)} FAILURE(S):")
    for f in failed:
        print("  -", f)
    sys.exit(1)
print("all checks passed")