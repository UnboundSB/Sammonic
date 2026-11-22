/**
 * generate-panel — renders a script to audio with the TTS pipeline.
 *
 * POST /generate returns 202 with a job id; the sidecar streams chunk
 * heartbeats (`i`, `total`, `chars`, `secs`) and the shell's poller publishes
 * them here. On completion the `@@RESULT` payload carries the MP3 path and its
 * duration in seconds.
 *
 * Two rules this panel exists to enforce visually:
 *  - training and TTS share the GPU and must never run concurrently (the
 *    sidecar answers a second GPU request with 409, which we surface verbatim);
 *  - the script is read from disk by the sidecar, so the panel shows the path
 *    it will render — including when the editor hands one over — instead of
 *    pretending a browser-side buffer is the input.
 */

import { LitElement, html, css } from "lit";
import { customElement, state } from "lit/decorators.js";

import * as api from "./api";
import { store } from "./state";
import type { GenerateResult, JobStatus } from "./api";
import type { AppState } from "./state";

function generateResult(job: JobStatus | null): GenerateResult | null {
  if (job === null || job.result === null) return null;
  const result = job.result;
  if (typeof (result as GenerateResult).duration !== "number") return null;
  return result as GenerateResult;
}

/** mm:ss for a seconds value; hours are not a realistic session length here. */
function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

@customElement("generate-panel")
export class GeneratePanel extends LitElement {
  static override styles = css`
    :host { display: block; }
    section {
      background: var(--panel-bg, #1b1e24);
      border: 1px solid var(--panel-border, #2c313a);
      border-radius: 8px; padding: 12px 14px;
    }
    .head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
    h2 { font-size: 13px; margin: 0; letter-spacing: 0.06em; text-transform: uppercase; color: var(--fg-dim, #9aa3af); }
    .spacer { flex: 1; }
    .state { font-size: 12px; color: var(--fg-dim, #9aa3af); font-family: ui-monospace, monospace; }
    .state.running { color: var(--accent, #7cc7ff); }
    .state.done { color: var(--ok, #6fd08c); }
    .state.error, .state.cancelled { color: var(--error, #ef6a6a); }
    label { font-size: 12px; color: var(--fg-dim, #9aa3af); display: block; margin: 6px 0 2px; }
    input {
      width: 100%; box-sizing: border-box; background: #101216; color: var(--fg, #e6e8ec);
      border: 1px solid var(--panel-border, #2c313a); border-radius: 6px; padding: 6px 8px;
      font-family: ui-monospace, monospace; font-size: 12px;
    }
    .row { display: flex; gap: 8px; margin-top: 10px; }
    button {
      background: #262b33; color: var(--fg, #e6e8ec);
      border: 1px solid var(--panel-border, #2c313a); border-radius: 6px;
      padding: 5px 10px; font-size: 12px; cursor: pointer;
    }
    button:hover:not(:disabled) { border-color: var(--accent, #7cc7ff); }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    button.danger { color: var(--error, #ef6a6a); }
    .bar { height: 8px; background: #101216; border: 1px solid var(--panel-border, #2c313a); border-radius: 4px; margin: 10px 0 4px; overflow: hidden; }
    .bar > div { height: 100%; background: var(--accent, #7cc7ff); transition: width 0.3s; }
    .meta { font-size: 12px; color: var(--fg-dim, #9aa3af); font-family: ui-monospace, monospace; }
    pre {
      background: #101216; border: 1px solid var(--panel-border, #2c313a); border-radius: 6px;
      padding: 8px; font-size: 11px; line-height: 1.45; max-height: 200px; overflow: auto;
      white-space: pre-wrap; color: var(--fg-dim, #9aa3af); margin: 6px 0 0;
    }
    .warn { font-size: 12px; color: var(--warn, #e3b341); margin-top: 8px; }
    .handoff { font-size: 12px; color: var(--accent, #7cc7ff); margin-top: 6px; }
    .result { font-size: 13px; color: var(--ok, #6fd08c); margin-top: 6px; }
    .error-line { color: var(--error, #ef6a6a); font-size: 12px; }
  `;

  @state() private app: AppState = store.getState();

  private unsubscribe: (() => void) | null = null;

  override connectedCallback(): void {
    super.connectedCallback();
    this.unsubscribe = store.subscribe((next) => {
      this.app = next;
    });
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    this.unsubscribe?.();
    this.unsubscribe = null;
  }

  private get job(): JobStatus | null {
    return store.watchedJob("generate");
  }

  /**
   * Non-null while another GPU job holds the card (or this very job does, in
   * which case the card being ours is the expected state).
   */
  private gpuConflict(): string | null {
    const { gpuBusyWith, job } = this.app;
    if (gpuBusyWith === null) return null;
    if (job !== null && job.id === gpuBusyWith) return null;
    return `GPU is busy with ${gpuBusyWith} — training and TTS share the card and cannot run at the same time.`;
  }

  private async start(): Promise<void> {
    store.setBusy("generate", true);
    try {
      const started = await api.generate(this.app.scriptPath, this.app.referencePath);
      store.startWatching("generate", started.job);
    } catch (err) {
      store.setBusy("generate", false);
      store.setNotice({ level: "error", text: `generate: ${api.errorMessage(err)}` });
    }
  }

  private async cancelJob(): Promise<void> {
    const job = this.job;
    if (job === null) return;
    try {
      await api.cancel(job.id); // the poller picks up the `cancelled` state
    } catch (err) {
      store.setNotice({ level: "error", text: `cancel: ${api.errorMessage(err)}` });
    }
  }

  override render() {
    const { scriptPath, referencePath, handoff } = this.app;
    const job = this.job;
    const busy = this.app.busy.generate;
    const live = job !== null && !api.isTerminalJobState(job.state);
    const progress = job?.progress ?? null;
    const fraction = job?.progress_fraction ?? null;
    const pct = fraction === null ? 0 : Math.round(fraction * 100);
    const result = generateResult(job);
    const conflict = this.gpuConflict();

    return html`
      <section>
        <div class="head">
          <h2>Generate — TTS render</h2>
          <span class="spacer"></span>
          <span class="state ${job?.state ?? ""}">${job === null ? "idle" : `${job.state}${job.id ? ` · ${job.id}` : ""}`}</span>
        </div>

        <label for="script">Script path (the sidecar reads this file from disk)</label>
        <input
          id="script"
          .value=${scriptPath}
          @input=${(event: Event) =>
            store.setScriptPath((event.currentTarget as HTMLInputElement).value)}
        />

        <label for="reference">Reference clip (≥ 3 s; the sidecar probes its duration)</label>
        <input
          id="reference"
          .value=${referencePath}
          @input=${(event: Event) =>
            store.setReferencePath((event.currentTarget as HTMLInputElement).value)}
        />

        <div class="row">
          <button type="button" ?disabled=${busy || live || conflict !== null} @click=${() => void this.start()}>
            Render
          </button>
          <button type="button" class="danger" ?disabled=${!live} @click=${() => void this.cancelJob()}>
            Cancel
          </button>
        </div>

        ${conflict !== null ? html`<div class="warn">${conflict}</div>` : ""}

        ${handoff !== null
          ? html`<div class="handoff">
              Handed over: ${handoff.bytes} bytes / ${handoff.text.length} chars — path
              <code>${handoff.path}</code>
            </div>`
          : ""}

        ${job !== null && job.error !== null ? html`<div class="error-line">${job.error}</div>` : ""}

        <div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}">
          <div style="width: ${pct}%"></div>
        </div>
        <div class="meta">
          ${progress !== null && typeof progress.i === "number"
            ? `chunk ${progress.i}/${progress.total ?? "?"}${typeof progress.secs === "number" ? ` · last chunk ${progress.secs.toFixed(1)}s` : ""}`
            : pct > 0
              ? `${pct}%`
              : "idle"}
          ${fraction !== null ? ` · ${pct}%` : ""}
        </div>

        ${result !== null
          ? html`<div class="result">
              done — ${result.mp3} (${formatDuration(result.duration)}, ${result.duration.toFixed(1)}s)
            </div>`
          : ""}

        ${job !== null && job.log_tail.length > 0
          ? html`<pre tabindex="0">${job.log_tail.join("\n")}</pre>`
          : ""}

        <div class="warn">
          Training and TTS share the GPU and cannot run at the same time — a render started while
          the trainer holds the card is rejected with a 409.
        </div>
      </section>
    `;
  }
}