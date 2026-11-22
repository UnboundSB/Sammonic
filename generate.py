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

def load_paths():
    """Resolve inputs/outputs, honouring CLI overrides. The sidecar passes
    explicit paths; interactive runs get the project-root defaults."""
    script, ref, out_wav, out_mp3 = SCRIPT, REF_AUDIO, OUT_WAV, OUT_MP3
    argv = sys.argv[1:]
    for i in range(0, len(argv) - 1, 2):
        flag, value = argv[i], argv[i + 1]
        if flag == "--script":
            script = os.path.abspath(value)
        elif flag == "--reference":
            ref = os.path.abspath(value)
        elif flag == "--out-wav":
            out_wav = os.path.abspath(value)
        elif flag == "--out-mp3":
            out_mp3 = os.path.abspath(value)
    return script, ref, out_wav, out_mp3

def main():
    script, ref, out_wav, out_mp3 = load_paths()

    # ---- Load script as chunks ----
    with open(script, encoding="utf-8") as f:
        text = f.read()
    chunks_text = [c.strip() for c in text.split("\n\n") if c.strip()]
    if not chunks_text:
        log(f"ERROR: {script} is empty.")
        sys.exit(1)
    log(f"Loaded {len(chunks_text)} chunks from {script} ({len(text)} chars)")

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