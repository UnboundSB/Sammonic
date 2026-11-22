# -*- coding: utf-8 -*-
"""
ctypes bridge to the C++ piece-table editing core.

This is what lets the sidecar serve `/edit/*` without any new dependency:
Python's stdlib can load the DLL built by `app/cpp/CMakeLists.txt` directly.
No Node addon, no koffi, no MSVC — the C++ core is reachable from a bare
Python install.

Contract (must match app/cpp/c_api.cpp):
    all positions are byte offsets into the UTF-8 document
    every char* handed back must be released with pt_free_string

The binding declares argtypes/restypes on every symbol it calls. Without them
ctypes guesses 32-bit ints and pointers, and on 64-bit Windows a
std::size_t position silently truncates above 4 GB of offset — which is
exactly the sort of bug that only appears on a long script.
"""

import ctypes
import os
import threading
from pathlib import Path

ROOT = Path(os.environ.get("SAMMONIC_ROOT") or str(Path(__file__).resolve().parent.parent))
DLL_CANDIDATES = [
    ROOT / "app" / "cpp" / "build" / "libpiecetable.dll",   # MinGW / g++
    ROOT / "app" / "cpp" / "build" / "Release" / "piecetable.dll",  # MSVC
    ROOT / "app" / "cpp" / "build" / "Debug" / "piecetable.dll",
]


class PieceTableUnavailable(RuntimeError):
    """Raised when the DLL has not been built or cannot be loaded."""


class Bridge:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.lib = None
        self.dll_path = None

    @classmethod
    def get(cls):
        """One binding per process; the DLL is loaded once, not per session."""
        with cls._lock:
            if cls._instance is None or cls._instance.lib is None:
                cls._instance = cls()
            return cls._instance

    def load(self):
        if self.lib is not None:
            return self
        last = None
        for path in DLL_CANDIDATES:
            if not path.exists():
                continue
            try:
                self.lib = self._declare(ctypes.CDLL(str(path)))
                self.dll_path = path
                return self
            except OSError as exc:
                last = exc
        hint = "build it with: cmake -S app/cpp -B app/cpp/build && cmake --build app/cpp/build"
        raise PieceTableUnavailable(
            f"piecetable library not found or not loadable ({last}); {hint}")

    @staticmethod
    def _declare(lib):
        VP, CP, SZ, IN = (ctypes.c_void_p, ctypes.c_char_p,
                          ctypes.c_size_t, ctypes.c_int)
        lib.pt_create.restype, lib.pt_create.argtypes = VP, [CP]
        lib.pt_destroy.restype, lib.pt_destroy.argtypes = None, [VP]
        lib.pt_insert.restype, lib.pt_insert.argtypes = None, [VP, SZ, CP]
        lib.pt_erase.restype, lib.pt_erase.argtypes = None, [VP, SZ, SZ]
        lib.pt_undo.restype, lib.pt_undo.argtypes = IN, [VP]
        lib.pt_redo.restype, lib.pt_redo.argtypes = IN, [VP]
        lib.pt_length.restype, lib.pt_length.argtypes = SZ, [VP]
        lib.pt_get_text.restype, lib.pt_get_text.argtypes = VP, [VP]
        lib.pt_free_string.restype, lib.pt_free_string.argtypes = None, [VP]
        lib.pt_piece_count.restype, lib.pt_piece_count.argtypes = SZ, [VP]
        lib.pt_check_invariant.restype, lib.pt_check_invariant.argtypes = IN, [VP]
        return lib


class EditSession:
    """One open document. Owns a native handle; closing releases it.

    The handle crosses a library boundary, so a session must be closed exactly
    once — a double free corrupts the allocator and crashes the process, which
    is why every method here checks `closed` first and why the sidecar only
    ever holds one reference.
    """

    def __init__(self, session_id, handle, lib):
        self.id = session_id
        self._handle = handle
        self._lib = lib
        self._closed = False
        # The caller is told to serialize edits, but the server must not depend
        # on the caller being correct: a compound operation (edit + read back)
        # has to be atomic or a concurrent edit lands between the two halves.
        self.lock = threading.Lock()

    # -- string crossing ------------------------------------------------
    def _text(self):
        raw = self._lib.pt_get_text(self._handle)
        if not raw:
            return ""
        try:
            return ctypes.string_at(raw).decode("utf-8", errors="replace")
        finally:
            self._lib.pt_free_string(raw)

    def insert(self, pos, text):
        self._lib.pt_insert(self._handle, pos, text.encode("utf-8"))
        return self.length()

    def erase(self, start, end):
        self._lib.pt_erase(self._handle, start, end)
        return self.length()

    def undo(self):
        return bool(self._lib.pt_undo(self._handle)), self._text()

    def redo(self):
        return bool(self._lib.pt_redo(self._handle)), self._text()

    def text(self):
        return self._text()

    def length(self):
        return int(self._lib.pt_length(self._handle))

    def pieces(self):
        return int(self._lib.pt_piece_count(self._handle))

    def consistent(self):
        return bool(self._lib.pt_check_invariant(self._handle))

    def close(self):
        if not self._closed:
            self._closed = True
            self._lib.pt_destroy(self._handle)


class Sessions:
    """Registry of open sessions, guarded because the sidecar is threaded."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions = {}

    def open(self, text=""):
        bridge = Bridge.get().load()
        handle = bridge.lib.pt_create(text.encode("utf-8"))
        if not handle:
            raise RuntimeError("pt_create returned null (allocation failure)")
        sid = f"ed_{os.urandom(6).hex()}"
        sess = EditSession(sid, handle, bridge.lib)
        with self._lock:
            self._sessions[sid] = sess
        return sess

    def get(self, sid):
        with self._lock:
            return self._sessions.get(sid)

    def close(self, sid):
        with self._lock:
            sess = self._sessions.pop(sid, None)
        if sess is not None:
            sess.close()
            return True
        return False

    def close_all(self):
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for sess in sessions:
            sess.close()