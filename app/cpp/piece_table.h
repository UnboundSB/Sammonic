// Piece-table text engine: O(log n) edits on multi-MB scripts, undo/redo,
// append-only buffers so RAM stays flat no matter how big the script gets.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hypo {

enum Buffer : uint8_t { ORIG = 0, ADD = 1 };

struct Piece {
    Buffer buf;
    size_t start;   // offset into that buffer
    size_t length;  // chars in this piece
};

struct EditOp {
    size_t pos;     // absolute doc position
    size_t old_len; // chars removed (0 for pure insert)
    std::string text; // text inserted ("" for pure delete)
};

class PieceTable {
public:
    PieceTable() = default;
    explicit PieceTable(std::string original);

    // Core edits
    void insert(size_t pos, const std::string& text);
    void erase(size_t start, size_t end);

    // Queries
    std::string get_text() const;           // full document
    std::string slice(size_t start, size_t len) const;
    size_t length() const { return length_; }
    size_t piece_count() const { return pieces_.size(); }

    // Undo / redo
    bool undo();
    bool redo();

    // Raw buffer access (for the bridge)
    const std::string& original() const { return orig_; }
    const std::string& add() const { return add_; }
    const std::vector<Piece>& pieces() const { return pieces_; }

private:
    std::string orig_;
    std::string add_;
    std::vector<Piece> pieces_;
    size_t length_ = 0;

    std::vector<EditOp> undo_stack_;
    std::vector<EditOp> redo_stack_;

    // Find piece index containing absolute pos, and offset within it.
    // Returns index; out_offset set to chars before pos inside that piece.
    size_t locate(size_t pos, size_t& out_offset) const;
    // Index of the piece whose left boundary sits at pos (pieces_.size() when
    // pos == length_). Requires a split_at(pos) boundary and the invariant
    // sum(piece lengths) == length_.
    size_t boundary_index(size_t pos) const;
    // Split piece so that an edit boundary lands at pos. No-op if already boundary.
    void split_at(size_t pos);
    // Append text to the ADD buffer and place a piece at boundary pos.
    void place_add(size_t pos, const std::string& text);
    // Rebuild piece list between two boundaries (used by erase).
    void erase_range(size_t start, size_t end);
};

} // namespace hypo