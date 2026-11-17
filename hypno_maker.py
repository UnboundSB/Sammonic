# -*- coding: utf-8 -*-
"""
Hypno Maker - Chatterbox TTS session builder
Single-file app: reference management, script editor, linter, generation, MP3 export.
Requires: chatterbox-tts, torch, torchaudio, pyloudnorm, ffmpeg on PATH.
"""

import os
import re
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_FILE = os.path.join(APP_DIR, "script.txt")
REF_FILE = os.path.join(APP_DIR, "reference_audio.wav")
OUT_WAV = os.path.join(APP_DIR, "output.wav")
OUT_MP3 = os.path.join(APP_DIR, "output.mp3")

SILENCE_BETWEEN_PARAS = 0.35
TARGET_LUFS = -23.0

ALLOWED_TAGS = {"[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]",
                "[clear throat]", "[sniff]", "[groan]", "[shush]"}
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF\U0001F900-\U0001F9FF\U0000FE00-\U0000FE0F\U0000200D]+"
)


# ---------------------------------------------------------------- utilities
def probe_duration(path):
    """Duration in seconds via ffprobe, or None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1", path],
            capture_output=True, text=True, timeout=30)
        m = re.search(r"([0-9.]+)", out.stdout)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def lint_script(text):
    """Return (errors, warnings, info) for the Chatterbox Turbo format."""
    errors, warnings, info = [], [], []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    for i, p in enumerate(paras, 1):
        if len(p) > 1000:
            errors.append(f"Chunk {i}: {len(p)} chars - over the 1000 hard limit, split it")
        elif len(p) > 500:
            warnings.append(f"Chunk {i}: {len(p)} chars - works, but 200-500 is the sweet spot")
        elif len(p) < 200:
            warnings.append(f"Chunk {i}: only {len(p)} chars - thin, consider extending")

    tags = re.findall(r"\[[a-z ]+\]", text)
    illegal = sorted(set(tags) - ALLOWED_TAGS)
    if illegal:
        errors.append("Illegal bracketed tags (Chatterbox reads them aloud): " + ", ".join(illegal))
    else:
        info.append(f"Tags OK: {len(tags)} used, all legal")

    if EMOJI_RE.search(text):
        errors.append("Emoji/symbol characters found - strip them")

    if "<" in text and ">" in text:
        errors.append("SSML-style angle brackets found - not supported")

    if re.search(r"\((\*|whisper|sigh|pause)", text, re.I):
        errors.append("Stage directions in parentheses found - not readable")

    info.append(f"Chunks: {len(paras)} | Total: {len(text)} chars")
    if paras:
        info.append(f"Chunk range: {len(min(paras, key=len))}-{len(max(paras, key=len))} chars")
    est = len(text) / 11.9  # ~11.9 chars/sec observed for Turbo
    info.append(f"Estimated audio: ~{est/60:.1f} min")
    return errors, warnings, info


# ------------------------------------------------------------- generation
def run_generation(msg_queue, script_text, ref_path):
    """Worker-thread pipeline. Sends messages: ('log', s) / ('prog', done, total)."""
    try:
        import torch
        import torchaudio as ta
        import pyloudnorm as pyln
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        msg_queue.put(("log", f"Reading script: {len(script_text)} chars"))
        chunks = [c.strip() for c in script_text.split("\n\n") if c.strip()]
        if not chunks:
            msg_queue.put(("log", "ERROR: script is empty"))
            return
        msg_queue.put(("log", f"{len(chunks)} chunks to generate"))

        msg_queue.put(("log", "Loading Chatterbox Turbo model..."))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        sr = model.sr
        msg_queue.put(("log", f"Model ready on {device} (sr={sr})"))

        silence = torch.zeros(int(sr * SILENCE_BETWEEN_PARAS))

        def trim(wav):
            thr = wav.abs().max() * (10 ** (-40 / 20))
            idx = (wav.abs() > thr).nonzero(as_tuple=True)[0]
            return wav[idx[0]: idx[-1] + 1] if len(idx) else wav

        def norm(wav):
            meter = pyln.Meter(sr)
            ln = meter.integrated_loudness(wav.numpy())
            if ln == float("-inf"):
                return wav
            import pyloudnorm as p2
            return torch.from_numpy(p2.normalize.loudness(wav.numpy(), ln, TARGET_LUFS)).float()

        pieces = []
        for i, chunk in enumerate(chunks, 1):
            msg_queue.put(("log", f"[{i}/{len(chunks)}] {chunk[:45]}..."))
            wav = model.generate(chunk, audio_prompt_path=ref_path)
            if wav.dim() == 2:
                wav = wav.squeeze(0)
            wav = norm(trim(wav.cpu().float()))
            pieces.append(wav)
            if i < len(chunks):
                pieces.append(silence.clone())
            msg_queue.put(("prog", i, len(chunks)))

        full = torch.cat(pieces)
        peak = full.abs().max()
        if peak > 0.95:
            full = full * (0.95 / peak)
        ta.save(OUT_WAV, full.unsqueeze(0), sr)
        msg_queue.put(("log", f"Saved output.wav ({full.shape[0]/sr:.1f}s)"))

        msg_queue.put(("log", "Converting to MP3..."))
        subprocess.run(["ffmpeg", "-y", "-i", OUT_WAV,
                        "-codec:a", "libmp3lame", "-qscale:a", "2", OUT_MP3],
                       check=True, capture_output=True)
        msg_queue.put(("log", "DONE: output.mp3"))
    except Exception as e:
        msg_queue.put(("log", f"ERROR: {e}"))


# ------------------------------------------------------------------- GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hypno Maker - Chatterbox TTS Session Builder")
        self.geometry("860x780")
        self.minsize(700, 600)
        self.msg_queue = queue.Queue()
        self.worker = None
        self._build_ui()
        self.after(100, self._poll_queue)

    # ---- UI ----
    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}

        # Reference row
        ref_frame = ttk.LabelFrame(self, text="Reference Audio (voice clone, 10s+ WAV)")
        ref_frame.pack(fill="x", **pad)
        ttk.Button(ref_frame, text="Browse...", command=self._browse_ref).pack(side="left", padx=6, pady=6)
        self.ref_label = ttk.Label(ref_frame, text="reference_audio.wav (default)")
        self.ref_label.pack(side="left", padx=6)
        self.ref_dur = ttk.Label(ref_frame, text="duration: -")
        self.ref_dur.pack(side="right", padx=6)

        # Script editor
        script_frame = ttk.LabelFrame(self, text="Script (blank line = pause/new chunk)")
        script_frame.pack(fill="both", expand=True, **pad)
        self.text = scrolledtext.ScrolledText(script_frame, wrap="word", font=("Consolas", 10))
        self.text.pack(fill="both", expand=True, padx=6, pady=6)
        btn_row = ttk.Frame(script_frame)
        btn_row.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(btn_row, text="Load script.txt", command=self._load_script).pack(side="left")
        ttk.Button(btn_row, text="Save", command=self._save_script).pack(side="left", padx=6)
        ttk.Button(btn_row, text="Lint", command=self._lint).pack(side="left", padx=6)

        # Lint report
        lint_frame = ttk.LabelFrame(self, text="Lint Report")
        lint_frame.pack(fill="x", **pad)
        self.lint_box = tk.Text(lint_frame, height=6, wrap="word", font=("Consolas", 9),
                                bg="#1e1e1e", fg="#d4d4d4", state="disabled")
        self.lint_box.pack(fill="x", padx=6, pady=6)

        # Generate row
        gen_frame = ttk.Frame(self)
        gen_frame.pack(fill="x", **pad)
        self.gen_btn = ttk.Button(gen_frame, text="GENERATE", command=self._generate)
        self.gen_btn.pack(side="left")
        self.progress = ttk.Progressbar(gen_frame, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=10)
        self.eta_label = ttk.Label(gen_frame, text="idle")
        self.eta_label.pack(side="right")

        # Log
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_box = scrolledtext.ScrolledText(log_frame, height=10, wrap="word",
                                                 font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4")
        self.log_box.pack(fill="both", expand=True, padx=6, pady=6)
        self.log_box.tag_config("err", foreground="#f44747")

        self._load_script()
        self._refresh_ref()

    # ---- actions ----
    def _browse_ref(self):
        path = filedialog.askopenfilename(
            initialdir=APP_DIR, title="Select reference WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if path:
            self.ref_path = path
            self._refresh_ref()

    def _refresh_ref(self):
        path = getattr(self, "ref_path", REF_FILE)
        dur = probe_duration(path) if os.path.exists(path) else None
        self.ref_label.config(text=os.path.basename(path) + ("" if os.path.exists(path) else "  (MISSING)"))
        if dur is None:
            self.ref_dur.config(text="duration: -")
        else:
            flag = "OK" if dur >= 10 else "TOO SHORT (need 10s+)"
            self.ref_dur.config(text=f"duration: {dur:.1f}s  {flag}")

    def _load_script(self):
        if os.path.exists(SCRIPT_FILE):
            with open(SCRIPT_FILE, encoding="utf-8") as f:
                self.text.delete("1.0", "end")
                self.text.insert("1.0", f.read())
            self._log(f"Loaded {SCRIPT_FILE}")

    def _save_script(self):
        with open(SCRIPT_FILE, "w", encoding="utf-8") as f:
            f.write(self.text.get("1.0", "end-1c"))
        self._log("Saved script.txt")

    def _lint(self):
        errors, warnings, info = lint_script(self.text.get("1.0", "end-1c"))
        self.lint_box.config(state="normal")
        self.lint_box.delete("1.0", "end")
        for e in errors:
            self.lint_box.insert("end", "ERROR: " + e + "\n", "err")
        for w in warnings:
            self.lint_box.insert("end", "warn:  " + w + "\n")
        for i in info:
            self.lint_box.insert("end", "info:  " + i + "\n")
        if not errors and not warnings:
            self.lint_box.insert("end", "All checks passed.\n")
        self.lint_box.config(state="disabled")

    def _generate(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Generation already running.")
            return
        self._save_script()
        ref = getattr(self, "ref_path", REF_FILE)
        if not os.path.exists(ref):
            messagebox.showerror("Missing reference", "Select a reference WAV first.")
            return
        text = self.text.get("1.0", "end-1c")
        self.gen_btn.config(state="disabled")
        self.eta_label.config(text="starting...")
        self.worker = threading.Thread(
            target=run_generation, args=(self.msg_queue, text, ref), daemon=True)
        self.worker.start()

    # ---- queue ----
    def _poll_queue(self):
        try:
            while True:
                kind, *rest = self.msg_queue.get_nowait()
                if kind == "log":
                    self._log(rest[0])
                elif kind == "prog":
                    done, total = rest
                    self.progress["value"] = done / total * 100
                    self.eta_label.config(text=f"{done}/{total} chunks")
                    if done == total:
                        self.gen_btn.config(state="normal")
                        self.eta_label.config(text="complete")
                        self._refresh_output()
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _refresh_output(self):
        dur = probe_duration(OUT_MP3)
        if dur:
            self._log(f"output.mp3: {dur:.1f}s ({dur/60:.1f} min)")

    def _log(self, msg):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")


if __name__ == "__main__":
    App().mainloop()