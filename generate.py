import sys
import subprocess
import torch
import torchaudio as ta
import pyloudnorm as pyln
from chatterbox.tts_turbo import ChatterboxTurboTTS

SCRIPT = "script.txt"
REF_AUDIO = "reference_audio.wav"
OUT_WAV = "output.wav"
OUT_MP3 = "output.mp3"
TARGET_LUFS = -23.0          # loudness target per chunk (matches seams)
SILENCE_BETWEEN_PARAS = 0.35  # seconds of silence between chunks
TRIM_TOP_DB = 40             # silence trim threshold

def log(msg):
    print(msg, flush=True)

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

def main():
    # ---- Load script as chunks ----
    with open(SCRIPT, encoding="utf-8") as f:
        text = f.read()
    chunks_text = [c.strip() for c in text.split("\n\n") if c.strip()]
    if not chunks_text:
        log("ERROR: script.txt is empty.")
        sys.exit(1)
    log(f"Loaded {len(chunks_text)} chunks from {SCRIPT} ({len(text)} chars)")

    # ---- Load model once ----
    log("Loading Chatterbox Turbo model...")
    model = ChatterboxTurboTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
    sr = model.sr
    log(f"Model ready (sample rate: {sr})")

    # ---- Generate, trim, normalize each chunk ----
    silence = torch.zeros(int(sr * SILENCE_BETWEEN_PARAS))
    pieces = []
    for i, chunk in enumerate(chunks_text, 1):
        log(f"[{i}/{len(chunks_text)}] '{chunk[:40]}...' ({len(chunk)} chars)")
        wav = model.generate(chunk, audio_prompt_path=REF_AUDIO)
        if wav.dim() == 2:
            wav = wav.squeeze(0)
        wav = wav.cpu().float()
        wav = trim_silence(wav, sr)
        wav = normalize_chunk(wav, sr)
        pieces.append(wav)
        if i < len(chunks_text):
            pieces.append(silence.clone())
        log(f"    -> {wav.shape[0] / sr:.1f}s after trim")

    # ---- Stitch ----
    full = torch.cat(pieces)
    full = peak_guard(full)
    ta.save(OUT_WAV, full.unsqueeze(0), sr)
    log(f"Saved {OUT_WAV} ({full.shape[0] / sr:.1f}s total)")

    # ---- Convert to MP3 ----
    log("Converting to MP3...")
    subprocess.run([
        "ffmpeg", "-y", "-i", OUT_WAV,
        "-codec:a", "libmp3lame", "-qscale:a", "2",
        OUT_MP3,
    ], check=True)
    log(f"DONE: {OUT_MP3}")

if __name__ == "__main__":
    main()