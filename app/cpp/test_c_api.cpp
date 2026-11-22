// Tests the exported C ABI, not the C++ class. This is the surface Tauri and
// the Node bridge bind to, so it gets its own harness: a bug that only
// manifests across the boundary (bad freeing, wrong handle semantics, a
// symbol that does not export) must fail here rather than in production.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
void* pt_create(const char* initial_text);
void pt_destroy(void* handle);
void pt_insert(void* handle, size_t pos, const char* text);
void pt_erase(void* handle, size_t start, size_t end);
int pt_undo(void* handle);
int pt_redo(void* handle);
size_t pt_length(void* handle);
char* pt_get_text(void* handle);
void pt_free_string(char* s);
size_t pt_piece_count(void* handle);
int pt_check_invariant(void* handle);
}

static int failures = 0;
static int checks = 0;

static void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("FAIL [%d] %s\n", checks, what.c_str());
    }
}

// RAII wrapper so a failed assertion still releases the handle.
struct Handle {
    void* h;
    explicit Handle(void* raw) : h(raw) {}
    ~Handle() { pt_destroy(h); }
};

static std::string text_of(void* h) {
    char* raw = pt_get_text(h);
    if (raw == nullptr) return std::string("<null>");
    std::string s(raw);
    pt_free_string(raw);
    return s;
}

int main() {
    // --- construction -------------------------------------------------
    {
        Handle h(pt_create("hello"));
        check(h.h != nullptr, "create returns a handle");
        check(pt_length(h.h) == 5, "length after create");
        check(text_of(h.h) == "hello", "text after create");
        check(pt_check_invariant(h.h) == 1, "invariant after create");
        check(pt_piece_count(h.h) == 1, "single piece after create");
    }
    {
        Handle h(pt_create(nullptr)); // must not crash
        check(h.h != nullptr, "create(nullptr) still returns a handle");
        check(pt_length(h.h) == 0, "create(nullptr) is empty");
    }
    {
        Handle h(pt_create(""));
        check(pt_length(h.h) == 0, "create(empty) is empty");
    }

    // --- null handles are inert, not fatal ---------------------------
    {
        check(pt_length(nullptr) == 0, "pt_length(nullptr) is 0");
        char* t = pt_get_text(nullptr);
        check(t != nullptr && std::strcmp(t, "") == 0, "pt_get_text(nullptr) is empty string");
        pt_free_string(t);
        pt_insert(nullptr, 0, "x");   // must return, not crash
        pt_erase(nullptr, 0, 1);
        check(pt_undo(nullptr) == 0, "pt_undo(nullptr) is 0");
        check(pt_redo(nullptr) == 0, "pt_redo(nullptr) is 0");
        check(pt_piece_count(nullptr) == 0, "pt_piece_count(nullptr) is 0");
        check(pt_check_invariant(nullptr) == 0, "pt_check_invariant(nullptr) is 0");
        pt_free_string(nullptr); // must be a no-op
        pt_destroy(nullptr);     // must be a no-op
    }

    // --- the regression suite, across the ABI ------------------------
    {
        Handle h(pt_create("0123456789"));
        pt_insert(h.h, 10, "ABC");
        check(pt_check_invariant(h.h) == 1, "invariant after end insert");
        check(text_of(h.h) == "0123456789ABC", "text after end insert");

        pt_erase(h.h, 0, pt_length(h.h));
        check(text_of(h.h) == "", "erase whole doc across ABI");
        check(pt_length(h.h) == 0, "length zero after full erase");
        check(pt_piece_count(h.h) == 0, "no pieces after full erase");
        check(pt_check_invariant(h.h) == 1, "invariant after full erase");
    }
    {
        Handle h(pt_create("0123456789"));
        pt_erase(h.h, 7, 10); // erase to end
        check(text_of(h.h) == "0123456", "erase to end across ABI");
        check(pt_check_invariant(h.h) == 1, "invariant after erase-to-end");
    }
    {
        Handle h(pt_create("0123456789"));
        pt_insert(h.h, 5, "abc");
        pt_erase(h.h, 0, 8);
        check(text_of(h.h) == "56789", "erase across boundary across ABI");
        check(pt_check_invariant(h.h) == 1, "invariant after cross-boundary erase");
    }

    // --- undo / redo across the ABI ----------------------------------
    {
        Handle h(pt_create("base"));
        pt_insert(h.h, 4, "XYZ");
        check(text_of(h.h) == "baseXYZ", "abi: before undo");
        check(pt_undo(h.h) == 1, "abi: undo returns 1");
        check(text_of(h.h) == "base", "abi: after undo");
        check(pt_redo(h.h) == 1, "abi: redo returns 1");
        check(text_of(h.h) == "baseXYZ", "abi: after redo");
        check(pt_undo(h.h) == 1, "abi: undo again");
        pt_insert(h.h, 4, "Q");
        check(pt_redo(h.h) == 0, "abi: redo stack cleared by new edit");
        check(text_of(h.h) == "baseQ", "abi: new edit applied");
        check(pt_undo(h.h) == 1, "abi: undo new edit");
        check(text_of(h.h) == "base", "abi: back to base");
        check(pt_undo(h.h) == 0, "abi: undo on empty stack returns 0");
        check(text_of(h.h) == "base", "abi: failed undo left text intact");
    }

    // --- multi-byte across the ABI -----------------------------------
    {
        Handle h(pt_create("\xC3\xA9\xE2\x82\xAC")); // é€ = 5 bytes
        check(pt_length(h.h) == 5, "abi: utf8 byte length 5");
        pt_erase(h.h, 0, 2);
        check(text_of(h.h) == std::string("\xE2\x82\xAC"), "abi: utf8 erase");
        pt_insert(h.h, 0, "\xC3\xA9");
        check(text_of(h.h) == std::string("\xC3\xA9\xE2\x82\xAC"), "abi: utf8 insert");
        check(pt_check_invariant(h.h) == 1, "abi: utf8 invariant");
    }

    // --- stress: many handles alive at once --------------------------
    {
        std::vector<void*> hs;
        for (int i = 0; i < 64; ++i) {
            void* h = pt_create("seed");
            for (int j = 0; j < 200; ++j) {
                std::string s = std::to_string(j);
                pt_insert(h, pt_length(h), s.c_str());
            }
            hs.push_back(h);
        }
        bool all_ok = true;
        std::string want = "seed";
        for (int j = 0; j < 200; ++j) want += std::to_string(j);
        for (void* h : hs) {
            if (text_of(h) != want || pt_check_invariant(h) != 1) { all_ok = false; break; }
        }
        check(all_ok, "64 concurrent handles stay independent and consistent");
        for (void* h : hs) pt_destroy(h);
    }

    // --- scale across the ABI ----------------------------------------
    {
        std::string big;
        for (int i = 0; i < 20000; ++i) big += "abcd ";
        Handle h(pt_create(big.c_str()));
        check(pt_length(h.h) == big.size(), "abi: 20k doc length");
        pt_insert(h.h, 10000, "NEEDLE");
        std::string want = big;
        want.insert(10000, "NEEDLE");
        check(text_of(h.h) == want, "abi: 20k doc insert");
        pt_erase(h.h, 10000, 10006);
        check(text_of(h.h) == big, "abi: 20k doc erase");
        check(pt_check_invariant(h.h) == 1, "abi: 20k doc invariant");
        // Piece count must stay bounded, not grow with document size.
        check(pt_piece_count(h.h) < 10, "abi: 20k doc piece count bounded (got " +
              std::to_string(pt_piece_count(h.h)) + ")");
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}