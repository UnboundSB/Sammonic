import re

path = "script.txt"

with open(path, encoding="utf-8") as f:
    text = f.read()

# Show what's there first
import unicodedata
found = {}
for ch in text:
    if ord(ch) > 127:
        found[ch] = found.get(ch, 0) + 1
for ch, n in sorted(found.items(), key=lambda x: ord(x[0])):
    if ord(ch) > 127:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "UNKNOWN"
        # print codepoint only, never the char itself (console is cp1252)
        print(f"U+{ord(ch):04X} x{n} ({name})")

# Strip emoji + other symbol ranges (keep letters, punctuation, typographic quotes/dashes)
emoji_pattern = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # emoji blocks
    "\U00002600-\U000027BF"   # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"   # flags
    "\U00002B00-\U00002BFF"   # arrows/symbols
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0000200D"              # ZWJ
    "\U000024C2-\U0001F251"
    "]+"
)

cleaned = emoji_pattern.sub("", text)

with open(path, "w", encoding="utf-8") as f:
    f.write(cleaned)

removed = len(text) - len(cleaned)
print(f"\nRemoved {removed} emoji characters.")

# Verify none left
with open(path, encoding="utf-8") as f:
    after = f.read()
leftover = sorted(set(ch for ch in after if ord(ch) > 127))
print("Remaining non-ASCII:", [f"U+{ord(c):04X} {c!r}" for c in leftover])