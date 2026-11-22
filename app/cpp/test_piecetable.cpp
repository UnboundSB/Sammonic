// Test harness for the piece-table engine + C ABI.
//
// Every mutation is followed by three checks:
//   1. invariant — the piece list sums to the reported length
//   2. content    — get_text() matches the expected string
//   3. ops        — piece_count stays bounded (coalescing still working)
// A failure in any of the three fails the whole suite; the editor relies on
// all three simultaneously.

#include "piece_table.h"

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;
static int checks = 0;

static void check(bool ok, const std::string& what, const hypo::PieceTable* /*pt*/ = nullptr) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("FAIL [%d] %s\n", checks, what.c_str());
    }
}

static hypo::PieceTable make(const std::string& text) {
    return hypo::PieceTable(std::string(text));
}

static void expect(const hypo::PieceTable& pt, const std::string& want, const char* what) {
    check(pt.get_text() == want, std::string(what) + ": text mismatch, got \"" +
          pt.get_text() + "\" want \"" + want + "\"", &pt);
    size_t sum = 0;
    for (const auto& p : pt.pieces()) sum += p.length;
    check(sum == pt.length(), std::string(what) + ": invariant broken", &pt);
    check(pt.length() == want.size(), std::string(what) + ": length mismatch", &pt);
}

int main() {
    // --- construction -------------------------------------------------
    {
        auto pt = make("");
        expect(pt, "", "empty construct");
        check(pt.piece_count() == 0, "empty construct: no pieces");
        pt.insert(0, "x");
        expect(pt, "x", "insert into empty at 0");
        pt.insert(99, "y"); // past end -> must clamp, not crash
        expect(pt, "xy", "insert past end clamps");
    }
    {
        auto pt = make("hello world");
        expect(pt, "hello world", "construct");
        check(pt.piece_count() == 1, "construct: single piece");
    }

    // --- insertion positions ------------------------------------------
    {
        auto pt = make("abcdef");
        pt.insert(0, "START ");
        expect(pt, "START abcdef", "insert at 0");
        // "START abcdef": index 6 is 'a'. Inserting at an existing piece
        // boundary must not shift the following text.
        pt.insert(6, "-");
        expect(pt, "START -abcdef", "insert at boundary");
        pt.insert(7, "MIDDLE ");
        expect(pt, "START -MIDDLE abcdef", "insert in middle");
        pt.insert(pt.length(), "END");
        expect(pt, "START -MIDDLE abcdefEND", "insert at end");
    }

    // --- insertion coalescing -----------------------------------------
    {
        auto pt = make("base");
        for (char c : std::string("abcdefghijklmnopqrstuvwxyz"))
            pt.insert(pt.length(), std::string(1, c));
        expect(pt, "baseabcdefghijklmnopqrstuvwxyz", "per-character append");
        check(pt.piece_count() <= 3, "per-character append: coalescing (got " +
              std::to_string(pt.piece_count()) + " pieces)");
    }
    {
        // Typing in the middle must not glob pieces together across a gap.
        // Inserting at a fixed position repeatedly reverses order — the
        // correct editor semantic, documented rather than assumed away.
        auto pt = make("XY");
        for (char c : std::string("abcd")) pt.insert(1, std::string(1, c));
        expect(pt, "XdcbaY", "middle typing reverses");
        auto pt2 = make("XY");
        for (char c : std::string("abcd")) pt2.insert(1 + pt2.length() - 2, std::string(1, c));
        expect(pt2, "XabcdY", "middle typing in order");
    }

    // --- erasure -------------------------------------------------------
    {
        auto pt = make("0123456789");
        pt.erase(3, 6); // removes indices 3,4,5 = "345"
        expect(pt, "0126789", "erase middle");
    }
    {
        // Regression: erasing to end-of-document must drop the trailing piece.
        auto pt = make("0123456789");
        pt.erase(7, 10);
        expect(pt, "0123456", "erase to end");
    }
    {
        // Regression: erasing the whole document.
        auto pt = make("0123456789");
        pt.erase(0, 10);
        expect(pt, "", "erase whole doc");
        check(pt.piece_count() == 0, "erase whole doc: no pieces left");
    }
    {
        // Regression: erase whole doc when there are ADD pieces at the end.
        auto pt = make("0123456789");
        pt.insert(10, "ABC");
        expect(pt, "0123456789ABC", "insert at end before full erase");
        pt.erase(0, pt.length());
        expect(pt, "", "erase whole doc with trailing ADD");
        check(pt.piece_count() == 0, "erase whole doc w/ ADD: no pieces left");
    }
    {
        // Multi-piece erase: remove a span crossing ORIG and ADD pieces.
        auto pt = make("0123456789");
        pt.insert(5, "abc"); // "01234abc56789"
        pt.erase(0, 8);      // removes "01234abc", crossing into ADD
        expect(pt, "56789", "erase across piece boundary");
    }
    {
        auto pt = make("0123456789");
        pt.erase(5, 5); // no-op
        expect(pt, "0123456789", "erase zero width");
        pt.erase(7, 3); // reversed -> no-op
        expect(pt, "0123456789", "erase reversed range");
        pt.erase(100, 200); // out of range -> no-op
        expect(pt, "0123456789", "erase out of range");
    }

    // --- undo / redo ---------------------------------------------------
    {
        auto pt = make("base");
        pt.insert(4, "XYZ");
        expect(pt, "baseXYZ", "before undo insert");
        check(pt.undo(), "undo insert returns true");
        expect(pt, "base", "after undo insert");

        pt.erase(0, 4);
        expect(pt, "", "before undo erase");
        check(pt.undo(), "undo erase returns true");
        expect(pt, "base", "after undo erase");

        check(!pt.undo(), "undo on empty stack returns false");
        expect(pt, "base", "failed undo changed nothing");
    }
    {
        // A fresh edit must invalidate the redo stack.
        auto pt = make("base");
        pt.insert(4, "A");
        pt.undo();
        expect(pt, "base", "undone insert");
        pt.insert(4, "B");
        expect(pt, "baseB", "new edit after undo");
        check(!pt.redo(), "redo cleared by new edit");
        expect(pt, "baseB", "redo attempt changed nothing");
    }
    {
        // Long chain: 50 edits, undo all 50, redo all 50.
        auto pt = make("");
        std::string want;
        for (int i = 0; i < 50; ++i) {
            std::string s = "[" + std::to_string(i) + "]";
            pt.insert(pt.length(), s);
            want += s;
        }
        expect(pt, want, "50 inserts");
        for (int i = 0; i < 50; ++i) {
            check(pt.undo(), "undo step " + std::to_string(i));
        }
        expect(pt, "", "undone everything");
        for (int i = 0; i < 50; ++i) {
            check(pt.redo(), "redo step " + std::to_string(i));
        }
        expect(pt, want, "redone everything");
    }
    {
        // Undo of an insert at a position that shifts under later edits.
        // Traced by hand: "abcdefgh" ->+QQ@2 "abQQcdefgh" ->-a "bQQcdefgh"
        //                ->+ZZ@1 "bZZQQcdefgh"
        auto pt = make("abcdefgh");
        pt.insert(2, "QQ");
        pt.erase(0, 1);
        pt.insert(1, "ZZ");
        expect(pt, "bZZQQcdefgh", "stack of mixed edits");
        for (int i = 0; i < 3; ++i) check(pt.undo(), "mixed undo " + std::to_string(i));
        expect(pt, "abcdefgh", "mixed edit chain undone");
        for (int i = 0; i < 3; ++i) check(pt.redo(), "mixed redo " + std::to_string(i));
        expect(pt, "bZZQQcdefgh", "mixed edit chain redone");
    }

    // --- UTF-8 ---------------------------------------------------------
    // The ABI is byte-addressed; multi-byte content should not corrupt.
    // é = C3 A9 (2 bytes), € = E2 82 AC (3 bytes), total 5.
    {
        auto pt = make("\xC3\xA9\xE2\x82\xAC");
        check(pt.length() == 5, "utf8: byte length is 5, got " + std::to_string(pt.length()));
        pt.erase(0, 2); // drop just the e-acute
        std::string euro = pt.get_text();
        check(euro == std::string("\xE2\x82\xAC"),
              "utf8: byte-precise erase, bytes=[" +
              std::to_string((unsigned char)euro[0]) + "," +
              std::to_string((unsigned char)euro[1]) + "," +
              std::to_string((unsigned char)euro[2]) + "]");
        pt.insert(0, "\xC3\xA9");
        std::string both = pt.get_text();
        check(both.size() == 5 && both == std::string("\xC3\xA9\xE2\x82\xAC"),
              "utf8: byte-precise insert and round trip");
    }

    // --- scale ---------------------------------------------------------
    {
        auto pt = make("");
        std::string big;
        for (int i = 0; i < 20000; ++i) big += "abcd ";
        pt = make(big);
        expect(pt, big, "20k-doc construct");
        pt.insert(10000, "NEEDLE");
        std::string want = big;
        want.insert(10000, "NEEDLE");
        expect(pt, want, "20k-doc insert");
        pt.erase(10000, 10006);
        expect(pt, big, "20k-doc erase");
        check(pt.undo(), "20k-doc undo");
        expect(pt, want, "20k-doc undo restores");
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}