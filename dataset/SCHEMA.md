# Hypnosis Script Dataset — Schema & Sourcing Plan

## Target: 150–300 diverse scripts, each 2,000–40,000 characters
Why: a QLoRA 3B needs ~100+ examples to learn a format contract plus enough
thematic variety to generalize. Your current script.txt is the seed sample.

## Raw input format
Drop raw scripts as plain .txt files in `dataset/raw/`.
One file = one complete script. No filenames conventions needed.

## Format contract (what every script must satisfy)
Enforced by `build_dataset.py` — violators are quarantined, not trained:
- Paragraphs of 200–500 chars (1,000 max); blank line separates chunks
- Only these bracketed tags: [laugh] [chuckle] [sigh] [gasp] [cough]
  [clear throat] [sniff] [groan] [shush]
- Ellipses (...) for pauses — never [pause] or missing-space punctuation
- No SSML, no emojis, no (*stage directions*), no extra bracket words
- ALL CAPS allowed for emphasis; numbers/symbols spelled out

## Sourcing options (pick any combination)
1. **Persona generation loop** — your Perchance persona generates scripts;
   save each as dataset/raw/NN_theme.txt. Run 30–50 themes, keep the linter happy.
2. **Hugging Face datasets** — search: hypnosis, trance induction,
   erotic hypnosis, meditation scripts (e.g. on HF hub). Convert to .txt.
3. **Manual curation** — any scripts you already own in text form.

Diversity checklist across the corpus:
- induction styles: escalator, forest, stairwell, beach, void, space
- themes: relaxation, sissification, age-play framing, control/surrender,
  masculine reforging, sleep, confidence
- lengths: mix of 3-min (thin) and 25-min (full arc) sessions
- tag density: some tag-heavy, some tag-free

## Build
python build_dataset.py
Outputs:
- dataset/train.jsonl
- dataset/val.jsonl   (10% split)
- dataset/rejected/   (scripts that failed the linter, with reasons)