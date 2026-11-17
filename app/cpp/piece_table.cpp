#include "piece_table.h"

#include <cassert>

namespace hypo {

// Stack contract (both stacks):
//   entries describe FORWARD edits: insert(pos, text)  -> {pos, 0, text}
//                                  erase(pos, len, rm) -> {pos, len, rm}
//   undo() applies the inverse of the forward edit
//   redo() re-applies the forward edit
// Keeps undo/redo symmetric and trivially correct.

PieceTable::PieceTable(std::string original)
    : orig_(std::move(original)), length_(orig_.size()) {
    if (!orig_.empty()) {
        pieces_.push_back({ORIG, 0, orig_.size()});
    }
}

size_t PieceTable::locate(size_t pos, size_t& out_offset) const {
    // Linear scan; scripts stay in the hundreds-of-pieces range.
    size_t acc = 0;
    for (size_t i = 0; i < pieces_.size(); ++i) {
        if (pos < acc + pieces_[i].length || i + 1 == pieces_.size()) {
            out_offset = pos - acc;
            return i;
        }
        acc += pieces_[i].length;
    }
    out_offset = 0;
    return pieces_.empty() ? 0 : pieces_.size() - 1;
}

void PieceTable::split_at(size_t pos) {
    if (pos == 0 || pos >= length_) return;
    size_t off = 0;
    size_t idx = locate(pos, off);
    if (off == 0) return; // already a boundary
    Piece& p = pieces_[idx];
    Piece right{p.buf, p.start + off, p.length - off};
    p.length = off;
    pieces_.insert(pieces_.begin() + static_cast<long>(idx) + 1, right);
}

void PieceTable::insert(size_t pos, const std::string& text) {
    if (text.empty()) return;
    if (pos > length_) pos = length_;

    split_at(pos);
    size_t off = 0;
    size_t idx = locate(pos, off);
    // off > 0 only when pos == length_ (insert at very end)
    size_t ins_idx = (off == 0) ? idx : idx + 1;

    size_t add_start = add_.size();
    add_.append(text);

    // Coalesce with the preceding ADD piece (rapid-typing hot path)
    if (ins_idx > 0 && pieces_[ins_idx - 1].buf == ADD &&
        pieces_[ins_idx - 1].start + pieces_[ins_idx - 1].length == add_start) {
        pieces_[ins_idx - 1].length += text.size();
    } else {
        pieces_.insert(pieces_.begin() + static_cast<long>(ins_idx),
                       Piece{ADD, add_start, text.size()});
    }

    length_ += text.size();

    EditOp op{pos, 0, text};
    undo_stack_.push_back(op);
    redo_stack_.clear();
}

void PieceTable::erase_range(size_t start, size_t end) {
    if (start >= end || start >= length_) return;
    if (end > length_) end = length_;

    split_at(start);
    split_at(end);

    size_t off_s = 0, off_e = 0;
    size_t i0 = locate(start, off_s);
    size_t i1 = locate(end, off_e);

    pieces_.erase(pieces_.begin() + static_cast<long>(i0),
                  pieces_.begin() + static_cast<long>(i1));
    length_ -= (end - start);
}

void PieceTable::erase(size_t start, size_t end) {
    if (start >= end || start >= length_) return;
    if (end > length_) end = length_;

    // Capture removed text for undo before touching pieces.
    std::string removed = slice(start, end - start);
    erase_range(start, end);

    // Coalesce ADD pieces that just became adjacent.
    for (size_t i = 1; i < pieces_.size();) {
        if (pieces_[i - 1].buf == ADD && pieces_[i].buf == ADD &&
            pieces_[i - 1].start + pieces_[i - 1].length == pieces_[i].start) {
            pieces_[i - 1].length += pieces_[i].length;
            pieces_.erase(pieces_.begin() + static_cast<long>(i));
        } else {
            ++i;
        }
    }

    undo_stack_.push_back({start, removed.size(), removed});
    redo_stack_.clear();
}

std::string PieceTable::slice(size_t start, size_t len) const {
    if (start >= length_ || len == 0) return {};
    if (start + len > length_) len = length_ - start;

    std::string out;
    out.reserve(len);
    size_t acc = 0;
    for (const Piece& p : pieces_) {
        if (acc + p.length <= start) { acc += p.length; continue; }
        size_t piece_begin = start > acc ? start - acc : 0;
        size_t take = p.length - piece_begin;
        if (take > len - out.size()) take = len - out.size();
        const std::string& src = (p.buf == ORIG) ? orig_ : add_;
        out.append(src, p.start + piece_begin, take);
        if (out.size() >= len) break;
        acc += p.length;
    }
    return out;
}

std::string PieceTable::get_text() const {
    return slice(0, length_);
}

bool PieceTable::undo() {
    if (undo_stack_.empty()) return false;
    EditOp op = undo_stack_.back();
    undo_stack_.pop_back();

    if (op.old_len == 0) {
        // forward was insert -> apply erase
        erase_range(op.pos, op.pos + op.text.size());
        redo_stack_.push_back(op); // forward op, ready to replay
    } else {
        // forward was erase -> apply re-insertion of removed text
        size_t add_start = add_.size();
        add_.append(op.text);
        split_at(op.pos);
        size_t off = 0;
        size_t idx = locate(op.pos, off);
        size_t ins_idx = (off == 0) ? idx : idx + 1;
        if (ins_idx > 0 && pieces_[ins_idx - 1].buf == ADD &&
            pieces_[ins_idx - 1].start + pieces_[ins_idx - 1].length == add_start) {
            pieces_[ins_idx - 1].length += op.text.size();
        } else {
            pieces_.insert(pieces_.begin() + static_cast<long>(ins_idx),
                           Piece{ADD, add_start, op.text.size()});
        }
        length_ += op.text.size();
        redo_stack_.push_back(op);
    }
    return true;
}

bool PieceTable::redo() {
    if (redo_stack_.empty()) return false;
    EditOp op = redo_stack_.back();
    redo_stack_.pop_back();

    if (op.old_len == 0) {
        // re-apply insert
        size_t add_start = add_.size();
        add_.append(op.text);
        split_at(op.pos);
        size_t off = 0;
        size_t idx = locate(op.pos, off);
        size_t ins_idx = (off == 0) ? idx : idx + 1;
        if (ins_idx > 0 && pieces_[ins_idx - 1].buf == ADD &&
            pieces_[ins_idx - 1].start + pieces_[ins_idx - 1].length == add_start) {
            pieces_[ins_idx - 1].length += op.text.size();
        } else {
            pieces_.insert(pieces_.begin() + static_cast<long>(ins_idx),
                           Piece{ADD, add_start, op.text.size()});
        }
        length_ += op.text.size();
    } else {
        // re-apply erase
        erase_range(op.pos, op.pos + op.old_len);
    }
    undo_stack_.push_back(op);
    return true;
}

} // namespace hypo