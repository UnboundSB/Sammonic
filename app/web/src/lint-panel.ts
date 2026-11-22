/**
 * lint-panel — the format gate, debounced.
 *
 * POSTs the current script (the editor's mirror in the store) to / lint with
 * a ~400 ms debounce, so a burst of keystrokes produces one request instead of
 * one per key. The linter is the same code the dataset builder uses
 * (single source of truth on the sidecar), so a pass here means a pass there.
 *
 * Rendering contract: errors block, warnings are "allowed if intentional", and
 * info carries statistics. A script with no errors shows an explicit
 * "passes" state — silence would be indistinguishable from "not run yet".
 */

import { LitElement, html, css } from "lit";
import { customElement, state } from "lit/decorators.js";

import * as api from "./api";
import { store } from "./state";
import type { LintResult } from "./api";

const DEBOUNCE_MS = 400;

@customElement("lint-panel")
export class LintPanel extends LitElement {
  static override styles = css`
    :host { display: block; }
    section {
      background: var(--panel-bg, #1b1e24);
      border: 1px solid var(--panel-border, #2c313a);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
    h2 { font-size: 13px; margin: 0; letter-spacing: 0.06em; text-transform: uppercase; color: var(--fg-dim, #9aa3af); }
    .spacer { flex: 1; }
    .badge { font-size: 12px; padding: 2px 10px; border-radius: 10px; border: 1px solid var(--panel-border, #2c313a); }
    .badge.pass { color: var(--ok, #6fd08c); border-color: var(--ok, #6fd08c); }
    .badge.fail { color: var(--error, #ef6a6a); border-color: var(--error, #ef6a6a); }
    .badge.idle { color: var(--fg-dim, #9aa3af); }
    .counts { display: flex; gap: 12px; font-size: 12px; margin: 6px 0; }
    .counts .e { color: var(--error, #ef6a6a); }
    .counts .w { color: var(--warn, #e3b341); }
    .counts .i { color: var(--fg-dim, #9aa3af); }
    ul { margin: 4px 0 8px; padding-left: 18px; font-size: 12px; }
    li { margin: 2px 0; }
    li.error { color: var(--error, #ef6a6a); }
    li.warning { color: var(--warn, #e3b341); }
    li.info { color: var(--fg-dim, #9aa3af); }
    button {
      background: #262b33; color: var(--fg, #e6e8ec);
      border: 1px solid var(--panel-border, #2c313a); border-radius: 6px;
      padding: 5px 10px; font-size: 12px; cursor: pointer;
    }
    button:hover:not(:disabled) { border-color: var(--accent, #7cc7ff); }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    .pending { color: var(--fg-dim, #9aa3af); font-size: 12px; }
  `;

  @state() private lint: LintResult | null = store.getState().lint;
  @state() private pending = false;
  @state() private lastRunLength = -1;

  private unsubscribe: (() => void) | null = null;
  private debounceTimer: number | null = null;
  private requestSeq = 0;

  override connectedCallback(): void {
    super.connectedCallback();
    this.unsubscribe = store.subscribe((next) => {
      this.lint = next.lint;
      this.onScriptChanged(next.scriptText);
    });
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    if (this.debounceTimer !== null) window.clearTimeout(this.debounceTimer);
    this.debounceTimer = null;
    this.unsubscribe?.();
    this.unsubscribe = null;
  }

  /**
   * One debounce slot: every script change resets the timer, and the request
   * that finally fires carries a sequence number so a slow earlier response
   * can never overwrite a newer one.
   */
  private onScriptChanged(text: string): void {
    if (this.debounceTimer !== null) window.clearTimeout(this.debounceTimer);
    if (text.trim().length === 0) {
      // Nothing to lint; clear the debounce and keep the previous result.
      this.pending = false;
      return;
    }
    this.pending = true;
    this.debounceTimer = window.setTimeout(() => {
      this.debounceTimer = null;
      void this.run();
    }, DEBOUNCE_MS);
  }

  private async run(): Promise<void> {
    const text = store.getState().scriptText;
    const seq = ++this.requestSeq;
    store.setBusy("lint", true);
    this.pending = true;
    try {
      const result = await api.lint(text);
      if (seq !== this.requestSeq) return; // superseded by a newer request
      store.setLint(result);
      this.lastRunLength = text.length;
    } catch (err) {
      if (seq !== this.requestSeq) return;
      store.setNotice({ level: "error", text: `lint: ${api.errorMessage(err)}` });
    } finally {
      if (seq === this.requestSeq) {
        this.pending = false;
        store.setBusy("lint", false);
      }
    }
  }

  override render() {
    const { lint } = this;

    return html`
      <section>
        <div class="head">
          <h2>Lint — format gate</h2>
          <span class="spacer"></span>
          <button type="button" ?disabled=${this.pending} @click=${() => void this.run()}>
            Run now
          </button>
          ${lint === null
            ? html`<span class="badge idle">not run</span>`
            : lint.ok
              ? html`<span class="badge pass">passes</span>`
              : html`<span class="badge fail">fails</span>`}
        </div>

        ${this.pending ? html`<div class="pending">linting…</div>` : ""}

        ${lint === null
          ? html`<div class="pending">Scripts are checked as you type; the linter is the same code the dataset builder runs.</div>`
          : html`
              <div class="counts">
                <span class="e">${lint.errors.length} error${lint.errors.length === 1 ? "" : "s"}</span>
                <span class="w">${lint.warnings.length} warning${lint.warnings.length === 1 ? "" : "s"}</span>
                <span class="i">${lint.info.length} note${lint.info.length === 1 ? "" : "s"}</span>
                <span class="i">(${this.lastRunLength} chars at last run)</span>
              </div>
              ${lint.ok && lint.warnings.length === 0
                ? html`<div class="pending" style="color: var(--ok, #6fd08c)">
                    passes — no problems found.
                  </div>`
                : ""}
              ${lint.errors.length > 0
                ? html`<ul>${lint.errors.map((e) => html`<li class="error">${e}</li>`)}</ul>`
                : ""}
              ${lint.warnings.length > 0
                ? html`<ul>${lint.warnings.map((w) => html`<li class="warning">${w}</li>`)}</ul>`
                : ""}
              ${lint.info.length > 0
                ? html`<ul>${lint.info.map((i) => html`<li class="info">${i}</li>`)}</ul>`
                : ""}
            `}
      </section>
    `;
  }
}