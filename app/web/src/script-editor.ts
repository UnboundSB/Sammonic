/**
 * script-editor — a textarea bound to the sidecar's piece-table session.
 *
 * Two invariants drive this component:
 *
 * 1. Every keystroke goes through the C++ core, not a JS string. The textarea
 *    is a *view* of the document; the document lives in `POST /edit/*` on the
 *    sidecar. The store's `scriptText` is only a mirror for the lint panel —
 *    no edit is ever computed by mutating a JS string and pushing the result.
 *    On each input event we diff the old and new view text and send the
 *    minimal op: one insert, one erase, or an erase-then-insert pair at the
 *    same offset.
 *
 * 2. All edit traffic for one session is serialized. The sidecar serves a
 *    session single-threaded: if edit #2 overtakes edit #1 on the wire, #2's
 *    byte offsets land in a document that has not seen #1 yet and the buffer
 *    is corrupted. So insert/erase/undo/redo/fetch all go through one promise
 *    chain, in the order the user produced them. `enqueue()` is that chain.
 *
 * Positions are BYTE offsets into the UTF-8 document (the sidecar's contract,
 * matching architecture.md §5.1), which is why the diff works in characters
 * and then converts — see `diffToOps`.
 */

import { LitElement, html, css } from "lit";
import { customElement, state } from "lit/decorators.js";

import * as api from "./api";
import { store } from "./state";

const encoder = new TextEncoder();

/** UTF-8 byte length of a JS string — the unit the sidecar counts in. */
function utf8ByteLength(text: string): number {
  return encoder.encode(text).length;
}

/**
 * One minimal edit, in BYTE offsets.
 *
 * A replace is expressed as erase-then-insert, both at the same start offset;
 * the sidecar applies them in that order, so the composition is exact.
 */
type EditOp =
  | { kind: "insert"; pos: number; text: string }
  | { kind: "erase"; start: number; end: number };

function commonPrefixLength(a: string, b: string): number {
  const max = Math.min(a.length, b.length);
  let i = 0;
  while (i < max && a.charCodeAt(i) === b.charCodeAt(i)) i += 1;
  return i;
}

function commonSuffixLength(a: string, b: string, prefix: number): number {
  // Never let the suffix reach into the prefix, or a full replacement would
  // produce overlapping (and therefore wrong) boundaries.
  const max = Math.min(a.length, b.length) - prefix;
  let s = 0;
  while (s < max && a.charCodeAt(a.length - 1 - s) === b.charCodeAt(b.length - 1 - s)) {
    s += 1;
  }
  return s;
}

/**
 * Minimal edit turning `prev` into `next`, as byte-offset sidecar ops.
 *
 * Why the diff walks characters and converts afterwards instead of diffing the
 * raw UTF-8 bytes: a byte-level prefix/suffix walk can stop in the middle of a
 * multi-byte character. The resulting "insert" would then be a byte slice that
 * is not valid UTF-8, and the sidecar would re-encode it as U+FFFD — silently
 * corrupting the document. Character-level boundaries are always valid UTF-8
 * boundaries, and converting two offsets to byte offsets is mechanical. The
 * cost is O(n) UTF-8 work per keystroke (three prefix slices); at script sizes
 * this tool targets (tens of KB) that is well under a millisecond, and the
 * correctness is not subtle.
 */
function diffToOps(prev: string, next: string): EditOp[] {
  const prefix = commonPrefixLength(prev, next);
  const suffix = commonSuffixLength(prev, next, prefix);

  const ops: EditOp[] = [];
  const delStart = prefix;
  const delEnd = prev.length - suffix;
  if (delEnd > delStart) {
    ops.push({
      kind: "erase",
      start: utf8ByteLength(prev.slice(0, delStart)),
      end: utf8ByteLength(prev.slice(0, delEnd)),
    });
  }
  const inserted = next.slice(prefix, next.length - suffix);
  if (inserted.length > 0) {
    ops.push({
      kind: "insert",
      pos: utf8ByteLength(prev.slice(0, prefix)),
      text: inserted,
    });
  }
  return ops;
}

@customElement("script-editor")
export class ScriptEditor extends LitElement {
  static override styles = css`
    :host { display: block; }
    section {
      background: var(--panel-bg, #1b1e24);
      border: 1px solid var(--panel-border, #2c313a);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin-bottom: 8px; }
    h2 { font-size: 13px; margin: 0; letter-spacing: 0.06em; text-transform: uppercase; color: var(--fg-dim, #9aa3af); }
    .spacer { flex: 1; }
    .readout { font-family: ui-monospace, monospace; font-size: 12px; color: var(--fg-dim, #9aa3af); }
    .readout b { color: var(--fg, #e6e8ec); font-weight: 600; }
    button {
      background: #262b33; color: var(--fg, #e6e8ec);
      border: 1px solid var(--panel-border, #2c313a); border-radius: 6px;
      padding: 5px 10px; font-size: 12px; cursor: pointer;
    }
    button:hover:not(:disabled) { border-color: var(--accent, #7cc7ff); }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    button.primary { border-color: var(--accent, #7cc7ff); color: var(--accent, #7cc7ff); }
    textarea {
      width: 100%; min-height: 280px; box-sizing: border-box;
      background: #101216; color: var(--fg, #e6e8ec);
      border: 1px solid var(--panel-border, #2c313a); border-radius: 6px;
      padding: 10px; font-family: ui-monospace, monospace; font-size: 13px;
      line-height: 1.5; resize: vertical;
    }
    textarea[readonly] { opacity: 0.75; }
    .status { font-size: 12px; margin-top: 6px; color: var(--fg-dim, #9aa3af); }
    .status.error { color: var(--error, #ef6a6a); }
    .status.ok { color: var(--ok, #6fd08c); }
    .help { font-size: 12px; color: var(--fg-dim, #9aa3af); margin-top: 6px; max-width: 90ch; }
  `;

  @state() private status: "opening" | "ready" | "error" = "opening";
  @state() private sessionId: string | null = null;
  @state() private bytes = 0;
  @state() private chars = 0;
  @state() private pieces = 0;
  @state() private canRedo = false;
  @state() private errorText: string | null = null;
  /** The sidecar's per-op invariant check (piece lengths sum to doc length). */
  @state() private consistent = true;

  /**
   * The document as the piece table currently reflects it — the baseline every
   * queued diff is computed against. NOT the same as the textarea's value while
   * edits are in flight.
   */
  private lastText = "";

  /** The serial edit chain. Every sidecar edit op is appended here. */
  private queue: Promise<unknown> = Promise.resolve();

  /** Number of links in the chain that have not settled yet. */
  private queued = 0;

  /**
   * True once an edit failed: further diffs would be computed against a
   * document that no longer matches the sidecar's, so we stop instead of
   * corrupting it and offer a re-open.
   */
  private desynced = false;

  /** Set while an undo/redo is in flight; see `enqueueControl`. */
  private controlInFlight = false;

  /** After the queue drains, pull authoritative text when this is set. */
  private resyncWanted = false;

  private alive = true;

  override connectedCallback(): void {
    super.connectedCallback();
    void this.openSession();
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    this.alive = false;
    const id = this.sessionId;
    this.sessionId = null;
    if (id !== null && !this.desynced) {
      // Best effort: a leaked session is the sidecar's to reap, and a rejected
      // close must not surface as a UI error during unmount.
      void api.editClose(id).catch(() => undefined);
    }
  }

  // -------------------------------------------------------------- session

  private async openSession(): Promise<void> {
    const seed = store.getState().scriptText;
    this.status = "opening";
    this.errorText = null;
    this.desynced = false;
    this.consistent = true;
    store.setBusy("editor", true);
    try {
      const opened = await api.editOpen(seed);
      if (!this.alive) return;
      this.sessionId = opened.session;
      this.lastText = seed;
      this.bytes = opened.length;
      this.chars = seed.length;
      this.pieces = opened.pieces;
      this.canRedo = false;
      this.status = "ready";
      this.setTextareaValue(seed);
      store.setScriptText(seed);
    } catch (err) {
      if (!this.alive) return;
      this.status = "error";
      this.errorText = api.errorMessage(err);
      store.setNotice({ level: "error", text: `editor: ${this.errorText}` });
    } finally {
      if (this.alive) store.setBusy("editor", false);
    }
  }

  private requireSession(): string {
    if (this.sessionId === null || this.desynced) {
      throw new Error("piece-table session unavailable — reopen the session");
    }
    return this.sessionId;
  }

  // ---------------------------------------------------------------- queue

  /**
   * Appends one sidecar edit to the serial chain and returns its promise.
   *
   * The chain is stored as `this.queue`; each link runs only after the
   * previous one settles (`then(task, task)` runs on both outcomes so a single
   * failure cannot wedge the chain forever). The chain itself never carries a
   * rejection — failures are reported through the returned promise's catch,
   * which `enqueue` attaches, so they reach `onEditFailure` exactly once.
   */
  private enqueue<T>(task: () => Promise<T>): Promise<T> {
    this.queued += 1;
    const run: Promise<T> = this.queue.then(task, task);
    this.queue = run.then(
      () => this.settle(),
      () => this.settle(),
    );
    run.catch((err: unknown) => this.onEditFailure(err));
    return run;
  }

  private settle(): void {
    this.queued -= 1;
    if (this.queued === 0 && this.resyncWanted) {
      this.resyncWanted = false;
      void this.resync();
    }
  }

  /**
   * Pulls the authoritative document back into the view.
   *
   * Needed after undo/redo: those rewrite a range wholesale, so the textarea's
   * content (and its caret offsets) are stale. This runs only once the queue
   * has drained, which is what makes the fetched text the true final state.
   */
  private async resync(): Promise<void> {
    try {
      const result = await api.editText(this.requireSession());
      if (!this.alive) return;
      this.lastText = result.text;
      this.bytes = result.length;
      this.chars = result.text.length;
      this.setTextareaValue(result.text);
      store.setScriptText(result.text);
      this.controlInFlight = false;
    } catch (err) {
      if (this.alive) this.onEditFailure(err);
    }
  }

  private onEditFailure(err: unknown): void {
    if (!this.alive) return;
    this.controlInFlight = false;
    if (this.desynced) {
      // First failure marks the session desynchronized; later queued links
      // only repeat it, so they are not re-reported.
      return;
    }
    this.desynced = true;
    this.status = "error";
    this.errorText = api.errorMessage(err);
    store.setNotice({ level: "error", text: `editor: ${this.errorText} — reopen the session to continue editing` });
  }

  // --------------------------------------------------------------- editing

  private get textarea(): HTMLTextAreaElement | null {
    return this.renderRoot.querySelector<HTMLTextAreaElement>("textarea");
  }

  private setTextareaValue(text: string): void {
    const textarea = this.textarea;
    if (textarea === null || textarea.value === text) return;
    textarea.value = text;
    // Caret to the end: undo/redo can rewrite a range the caret was inside,
    // and there is no meaningful position to restore from a full-document
    // replacement.
    textarea.selectionStart = text.length;
    textarea.selectionEnd = text.length;
  }

  private onInput(event: Event): void {
    const textarea = event.currentTarget as HTMLTextAreaElement;
    const next = textarea.value;
    if (this.controlInFlight) {
      // An undo/redo is mid-flight and will rewrite the document; a diff
      // computed now would use offsets the undo is about to invalidate. The
      // textarea is readOnly during the round trip, so this is a belt-and-
      // braces guard.
      return;
    }
    const prev = this.lastText;
    if (next === prev) return;

    this.lastText = next;
    this.chars = next.length;
    this.bytes = utf8ByteLength(next); // optimistic; the op response confirms
    store.setScriptText(next); // mirror for the lint panel only

    for (const op of diffToOps(prev, next)) {
      this.enqueue(() => this.applyOp(op));
    }
  }

  private async applyOp(op: EditOp): Promise<void> {
    const id = this.requireSession();
    if (op.kind === "erase") {
      const result = await api.editErase(id, op.start, op.end);
      this.bytes = result.length;
      this.pieces = result.pieces;
      this.consistent = result.consistent ?? true;
    } else {
      const result = await api.editInsert(id, op.pos, op.text);
      this.bytes = result.length;
      this.pieces = result.pieces;
      this.consistent = result.consistent ?? true;
    }
    // A fresh edit clears the sidecar's redo stack, so the button must not
    // offer a redo that would come back `redone: false`.
    this.canRedo = false;
  }

  /**
   * Queues an undo/redo. While one is in flight the textarea is readOnly,
   * because its offsets are relative to the document the control op is about
   * to rewrite; `resync()` re-enables editing with the authoritative text.
   */
  private enqueueControl(action: "undo" | "redo"): void {
    if (this.sessionId === null || this.desynced) return;
    this.controlInFlight = true;
    this.resyncWanted = true;
    void this.enqueue(async () => {
      const id = this.requireSession();
      if (action === "undo") {
        const result = await api.editUndo(id);
        if (!result.undone) return; // stack empty — the sidecar said so
        this.canRedo = true;
        this.lastText = result.text;
        this.bytes = result.length;
        this.chars = result.text.length;
      } else {
        const result = await api.editRedo(id);
        if (!result.redone) return;
        this.canRedo = false;
        this.lastText = result.text;
        this.bytes = result.length;
        this.chars = result.text.length;
      }
    });
  }

  // -------------------------------------------------------------- handoff

  /**
   * "Hand to generate" — no file is written from the browser (there is no fs
   * access here, and faking a save would only hide drift). The handoff carries
   * the repo-relative script path plus the current text; the sidecar resolves
   * relative paths against the project root and renders the file on disk, so
   * the path is the contract and the text is for display and linting. When a
   * reference clip is already configured the render starts immediately.
   */
  private handoff(): void {
    const text = this.lastText;
    const bytes = utf8ByteLength(text);
    const path = store.getState().scriptPath;
    store.setHandoff({ path, text, bytes });
    store.setNotice({
      level: "info",
      text: `Handed ${bytes} bytes (${text.length} chars) to the generate panel — path ${path}.`,
    });
    const reference = store.getState().referencePath.trim();
    if (reference.length > 0) void this.startGenerate(reference);
  }

  private async startGenerate(reference: string): Promise<void> {
    store.setBusy("generate", true);
    try {
      const started = await api.generate(store.getState().scriptPath, reference);
      store.startWatching("generate", started.job);
    } catch (err) {
      store.setBusy("generate", false);
      store.setNotice({ level: "error", text: api.errorMessage(err) });
    }
  }

  override render() {
    const busy = store.getState().busy.editor;
    const editing = this.status === "ready" && !this.controlInFlight;

    return html`
      <section>
        <div class="head">
          <h2>Script editor — piece table</h2>
          <button type="button" ?disabled=${!editing} @click=${() => this.enqueueControl("undo")}>
            Undo
          </button>
          <button type="button" ?disabled=${!editing || !this.canRedo} @click=${() => this.enqueueControl("redo")}>
            Redo
          </button>
          <span class="spacer"></span>
          <span class="readout" title="Byte length as the piece table counts it (UTF-8)">
            bytes <b>${this.bytes}</b> · chars <b>${this.chars}</b> · pieces <b>${this.pieces}</b>
            ${this.consistent ? "" : html` · <b style="color: var(--error, #ef6a6a)">inconsistent</b>`}
          </span>
          <button type="button" class="primary" ?disabled=${!editing} @click=${this.handoff}>
            Hand to generate
          </button>
        </div>

        <!--
          The value is never bound reactively: writing .value on every render
          would fight the browser's own caret handling while the user types.
          Content is set imperatively (initial seed, post-undo resync) only.
        -->
        <textarea
          placeholder="Type or paste a draft here. Every keystroke is applied to the piece-table buffer on the sidecar, not to a copy in the page."
          ?readonly=${!editing}
          spellcheck="false"
          @input=${this.onInput}
        ></textarea>

        ${this.status === "error"
          ? html`<div class="status error" role="alert">
              Session error: ${this.errorText ?? "unknown"} —
              <button type="button" ?disabled=${busy} @click=${() => void this.openSession()}>
                reopen session
              </button>
            </div>`
          : html`<div class="status ${this.controlInFlight ? "" : "ok"}">
              ${this.status === "opening"
                ? "opening piece-table session…"
                : this.controlInFlight
                  ? "applying undo/redo…"
                  : `session ${this.sessionId ?? "—"} · every edit serialized through one queue`}
            </div>`}

        <div class="help">
          Edits are sent as byte-offset insert/erase ops to <code>/edit/*</code>; the C++ piece
          table on the sidecar stays the single source of truth for the document. "Hand to
          generate" passes the current text and script path to the render panel — it does not
          write files; the sidecar renders the script file on disk.
        </div>
      </section>
    `;
  }
}