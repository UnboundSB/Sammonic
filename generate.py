import sys
import os
import json
import subprocess
import torch
import torchaudio as ta
import pyloudnorm as pyln
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Paths are resolved against the project root, not the current working
# directory, so the sidecar can spawn this from anywhere. SAMMONIC_ROOT wins
# when set; otherwise the directory this file lives in.
ROOT = os.environ.get("SAMMONIC_ROOT") or os.path.dirname(os.path.abspath(__file__))

SCRIPT = os.path.join(ROOT, "script.txt")
REF_AUDIO = os.path.join(ROOT, "reference_audio.wav")
OUT_WAV = os.path.join(ROOT, "output.wav")
OUT_MP3 = os.path.join(ROOT, "output.mp3")

TARGET_LUFS = -23.0          # loudness target per chunk (matches seams)
SILENCE_BETWEEN_PARAS = 0.35  # seconds of silence between chunks
TRIM_TOP_DB = 40             # silence trim threshold

def log(msg):
    print(msg, flush=True)

def emit_progress(i, total, chars, secs):
    """Heartbeat for the sidecar. Keep this line machine-readable: the sidecar
    parses it, humans read the [i/total] line instead."""
    print("@@PROGRESS " + json.dumps(
        {"i": i, "total": total, "chars": chars, "secs": round(secs, 2)}), flush=True)


def review(chunks_text, script):
    """Print the script that is about to be rendered, in the order it will be
    spoken, before any GPU work happens.

    A render takes about twenty minutes of GPU. Reading the text first costs
    nothing, and catching a wrong or stale script here is the difference
    between a review and a wasted hour. This is why it runs before the model
    is loaded rather than after.
    """
    sizes = [len(c) for c in chunks_text]
    print("\n" + "=" * 70)
    print(f"SCRIPT TO BE RENDERED: {script}")
    print(f"{len(chunks_text)} chunks | {sum(sizes)} chars | "
          f"chunk range {min(sizes)}-{max(sizes)}")
    print("=" * 70)
    for i, chunk in enumerate(chunks_text, 1):
        print(f"\n[chunk {i}/{len(chunks_text)}] ({len(chunk)} chars)")
        print(chunk)
    print("\n" + "=" * 70)
    print("END OF SCRIPT -- generation starts after this point.")
    print("=" * 70 + "\n", flush=True)

def emit_result(path, secs):
    print("@@RESULT " + json.dumps({"mp3": path, "duration": round(secs, 2)}), flush=True)

def trim_silence(wav, sr, top_db=TRIM_TOP_DB):
    """Trim leading/trailing near-silence from a 1-D tensor."""
    if wav.numel() == 0:
        return wav
    threshold = wav.abs().max() * (10 ** (-top_db / 20))
    idx = (wav.abs() > threshold).nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        return wav
    return wav[idx[0]: idx[-1] + 1]

def normalize_chunk(wav, sr, target_lufs=TARGET_LUFS):
    """Loudness-normalize one chunk to target LUFS."""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav.numpy())
    if loudness == float("-inf"):
        return wav
    return torch.from_numpy(
        pyln.normalize.loudness(wav.numpy(), loudness, target_lufs)
    ).float()

def peak_guard(wav, ceiling=0.95):
    """Prevent clipping after concatenation."""
    peak = wav.abs().max()
    if peak > ceiling:
        wav = wav * (ceiling / peak)
    return wav

def load_options():
    """Resolve CLI options: paths plus the review/preview switches.

    Argparse would normally do this, but the existing hand-rolled parsing is
    what the sidecar already calls with explicit --script/--reference, and
    switching parsers is a change the working audio path does not need.
    """
    opts = {
        "script": SCRIPT, "reference": REF_AUDIO,
        "out_wav": OUT_WAV, "out_mp3": OUT_MP3,
        "preview": False, "quiet": False,
    }
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        flag = argv[i]
        if flag == "--preview":
            opts["preview"] = True
        elif flag == "--quiet":
            opts["quiet"] = True
        elif flag.startswith("--") and i + 1 < len(argv):
            value = argv[i + 1]
            key = flag[2:].replace("-", "_")
            if key in ("script", "reference", "out_wav", "out_mp3"):
                opts[key] = os.path.abspath(value)
            i += 1
        i += 1
    return opts

def main():
    opts = load_options()
    script, ref = opts["script"], opts["reference"]
    out_wav, out_mp3 = opts["out_wav"], opts["out_mp3"]

    # ---- Load script as chunks ----
    with open(script, encoding="utf-8") as f:
        text = f.read()
    chunks_text = [c.strip() for c in text.split("\n\n") if c.strip()]
    if not chunks_text:
        log(f"ERROR: {script} is empty.")
        sys.exit(1)
    log(f"Loaded {len(chunks_text)} chunks from {script} ({len(text)} chars)")

    # ---- Show what is about to be rendered, before touching the GPU ----
    if not opts["quiet"]:
        review(chunks_text, script)
    if opts["preview"]:
        log("--preview given: script printed, no audio generated.")
        return

    # ---- Load model once ----
    log("Loading Chatterbox Turbo model...")
    model = ChatterboxTurboTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
    sr = model.sr
    log(f"Model ready (sample rate: {sr})")

    # ---- Generate, trim, normalize each chunk ----
    silence = torch.zeros(int(sr * SILENCE_BETWEEN_PARAS))
    pieces = []
    total = len(chunks_text)
    for i, chunk in enumerate(chunks_text, 1):
        log(f"[{i}/{total}] '{chunk[:40]}...' ({len(chunk)} chars)")
        wav = model.generate(chunk, audio_prompt_path=ref)
        if wav.dim() == 2:
            wav = wav.squeeze(0)
        wav = wav.cpu().float()
        wav = trim_silence(wav, sr)
        wav = normalize_chunk(wav, sr)
        pieces.append(wav)
        if i < total:
            pieces.append(silence.clone())
        secs = wav.shape[0] / sr
        log(f"    -> {secs:.1f}s after trim")
        emit_progress(i, total, len(chunk), secs)

    # ---- Stitch ----
    full = torch.cat(pieces)
    full = peak_guard(full)
    ta.save(out_wav, full.unsqueeze(0), sr)
    log(f"Saved {out_wav} ({full.shape[0] / sr:.1f}s total)")

    # ---- Convert to MP3 ----
    log("Converting to MP3...")
    subprocess.run([
        "ffmpeg", "-y", "-i", out_wav,
        "-codec:a", "libmp3lame", "-qscale:a", "2",
        out_mp3,
    ], check=True)
    log(f"DONE: {out_mp3}")
    emit_result(out_mp3, full.shape[0] / sr)

if __name__ == "__main__":
    main()