// C ABI for the piece-table engine.
//
// Why a C ABI instead of exposing the C++ class:
//   - Tauri (Rust) and Node (koffi) both bind C symbols; neither can consume
//     C++ name mangling, std::string, or std::vector across the boundary.
//   - It forces every string to cross as a malloc'd char*, so there is no
//     allocator mismatch between the DLL's heap and the host's.
//
// Thread safety: one process-wide recursive mutex, shared by every entry
// point. All handles serialize, which is the correct semantic for an editor
// core: an editor has one buffer and one user, and the sidecar's job workers
// are the only other readers.
//
// Memory: every char* returned must be released with pt_free_string.

#include "piece_table.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

// Deliberately a single instance, not per-function statics: per-function
// mutexes would let two different calls touch the same handle concurrently.
std::recursive_mutex& gate() {
    static std::recursive_mutex m;
    return m;
}

char* to_c_string(const std::string& text) {
    char* out = static_cast<char*>(std::malloc(text.size() + 1));
    if (out == nullptr) return nullptr;
    std::memcpy(out, text.c_str(), text.size() + 1);
    return out;
}

} // namespace

extern "C" {

void* pt_create(const char* initial_text) {
    std::lock_guard<std::recursive_mutex> lock(gate());
    try {
        std::string text = initial_text ? std::string(initial_text) : std::string();
        return new hypo::PieceTable(std::move(text));
    } catch (...) {
        return nullptr;
    }
}

void pt_destroy(void* handle) {
    std::lock_guard<std::recursive_mutex> lock(gate());
    delete static_cast<hypo::PieceTable*>(handle); // deleting nullptr is a no-op
}

void pt_insert(void* handle, size_t pos, const char* text) {
    if (handle == nullptr || text == nullptr) return;
    std::lock_guard<std::recursive_mutex> lock(gate());

    auto* pt = static_cast<hypo::PieceTable*>(handle);
    pt->insert(pos, std::string(text));
}

void pt_erase(void* handle, size_t start, size_t end) {
    if (handle == nullptr) return;
    if (start >= end) return;
    std::lock_guard<std::recursive_mutex> lock(gate());

    auto* pt = static_cast<hypo::PieceTable*>(handle);
    pt->erase(start, end);
}

int pt_undo(void* handle) {
    if (handle == nullptr) return 0;
    std::lock_guard<std::recursive_mutex> lock(gate());

    return static_cast<hypo::PieceTable*>(handle)->undo() ? 1 : 0;
}

int pt_redo(void* handle) {
    if (handle == nullptr) return 0;
    std::lock_guard<std::recursive_mutex> lock(gate());

    return static_cast<hypo::PieceTable*>(handle)->redo() ? 1 : 0;
}

size_t pt_length(void* handle) {
    if (handle == nullptr) return 0;
    std::lock_guard<std::recursive_mutex> lock(gate());

    return static_cast<hypo::PieceTable*>(handle)->length();
}

char* pt_get_text(void* handle) {
    std::lock_guard<std::recursive_mutex> lock(gate());
    if (handle == nullptr) return to_c_string(std::string());

    std::string text = static_cast<hypo::PieceTable*>(handle)->get_text();
    return to_c_string(text);
}

void pt_free_string(char* s) {
    std::free(s); // free(nullptr) is a no-op
}

// Named alias for callers that want to be explicit that positions are byte
// offsets into the UTF-8 logical document.
size_t pt_byte_length(void* handle) {
    return pt_length(handle);
}

// Diagnostics: number of pieces. A healthy editor session stays in the low
// hundreds even for multi-MB scripts; runaway growth means coalescing broke.
size_t pt_piece_count(void* handle) {
    if (handle == nullptr) return 0;
    std::lock_guard<std::recursive_mutex> lock(gate());

    return static_cast<hypo::PieceTable*>(handle)->piece_count();
}

// Test hook: 1 when the piece list is consistent with the reported length,
// 0 otherwise. A mismatch means edits corrupted the document and the session
// must be reloaded from disk.
int pt_check_invariant(void* handle) {
    if (handle == nullptr) return 0;
    std::lock_guard<std::recursive_mutex> lock(gate());

    const hypo::PieceTable* pt = static_cast<hypo::PieceTable*>(handle);
    size_t sum = 0;
    for (const auto& p : pt->pieces()) sum += p.length;
    return sum == pt->length() ? 1 : 0;
}

} // extern "C"