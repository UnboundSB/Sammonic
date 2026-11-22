/**
 * dataset-panel — runs the dataset build and reports its result.
 *
 * POST /dataset/build starts a CPU-only background job (202 Accepted); the
 * shell's single status poller (see hypno-app) then owns GET /status/:id and
 * publishes each snapshot into the store. This panel renders that snapshot
 * when it belongs to a `dataset` job — polling through one place means two
 * panels can never double-poll, and a job started elsewhere still shows up.
 *
 * On completion the sidecar's `@@RESULT` payload carries the counts the
 * builder prints (scripts / accepted / rejected / samples / train / val).
 */

import { LitElement, html, css } from "lit";
import { customElement, state } from "lit/decorators.js";

import * as api from "./api";
import { store } from "./state";
import type { DatasetResult, JobStatus } from "./api";
import type { AppState } from "./state";

/** Narrows a job result to the dataset shape, or null if it isn't one. */
function datasetResult(job: JobStatus | null): DatasetResult | null {
  if (job === null || job.result === null) return null;
  const result = job.result;
  if (typeof (result as DatasetResult).scripts !== "number") return null;
  return result as DatasetResult;
}

@customElement("dataset-panel")
export class DatasetPanel extends LitElement {
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
    .state { font-size: 12px; color: var(--fg-dim, #9aa3af); font-family: ui-monospace, monospace; }
    .state.running { color: var(--accent, #7cc7ff); }
    .state.done { color: var(--ok, #6fd08c); }
    .state.error, .state.cancelled { color: var(--error, #ef6a6a); }
    button {
      background: #262b33; color: var(--fg, #e6e8ec);
      border: 1px solid var(--panel-border, #2c313a); border-radius: 6px;
      padding: 5px 10px; font-size: 12px; cursor: pointer;
    }
    button:hover:not(:disabled) { border-color: var(--accent, #7cc7ff); }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    .counts { display: grid; grid-template-columns: repeat(auto-fit, minmax(90px, 1fr)); gap: 8px; margin: 10px 0; }
    .count { background: #14161a; border: 1px solid var(--panel-border, #2c313a); border-radius: 6px; padding: 8px; text-align: center; }
    .count b { display: block; font-size: 18px; }
    .count span { font-size: 11px; color: var(--fg-dim, #9aa3af); }
    .count.bad b { color: var(--error, #ef6a6a); }
    pre {
      background: #101216; border: 1px solid var(--panel-border, #2c313a); border-radius: 6px;
      padding: 8px; font-size: 11px; line-height: 1.45; max-height: 180px; overflow: auto;
      white-space: pre-wrap; color: var(--fg-dim, #9aa3af); margin: 6px 0 0;
    }
    .help { font-size: 12px; color: var(--fg-dim, #9aa3af); margin-top: 6px; }
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
    return store.watchedJob("dataset");
  }

  private async build(): Promise<void> {
    store.setBusy("dataset", true);
    try {
      // No raw_dir sent: the sidecar defaults to <project root>/dataset/raw
      // and seeds it from the bundled sample when it is empty.
      const started = await api.datasetBuild();
      store.startWatching("dataset", started.job);
    } catch (err) {
      store.setBusy("dataset", false);
      store.setNotice({ level: "error", text: `dataset: ${api.errorMessage(err)}` });
    }
  }

  override render() {
    const job = this.job;
    const busy = this.app.busy.dataset;
    const result = datasetResult(job);
    const live = job !== null && !api.isTerminalJobState(job.state);

    return html`
      <section>
        <div class="head">
          <h2>Dataset build</h2>
          <span class="spacer"></span>
          <span class="state ${job?.state ?? ""}">${job === null ? "idle" : `${job.state}${job.id ? ` · ${job.id}` : ""}`}</span>
          <button type="button" ?disabled=${busy || live} @click=${() => void this.build()}>
            Build dataset
          </button>
        </div>

        ${job !== null && job.error !== null
          ? html`<div class="error-line">${job.error}</div>`
          : ""}

        ${result !== null
          ? html`
              <div class="counts">
                <div class="count"><b>${result.scripts}</b><span>scripts</span></div>
                <div class="count"><b>${result.accepted}</b><span>accepted</span></div>
                <div class="count bad"><b>${result.rejected}</b><span>rejected</span></div>
                <div class="count"><b>${result.samples}</b><span>samples</span></div>
                <div class="count"><b>${result.train}</b><span>train</span></div>
                <div class="count"><b>${result.val}</b><span>val</span></div>
              </div>
            `
          : job !== null && job.progress !== null && typeof job.progress.i === "number"
            ? html`<div class="state running">script ${job.progress.i}/${job.progress.total ?? "?"}${job.progress.name ? ` — ${job.progress.name}` : ""}</div>`
            : ""}

        ${job !== null && job.log_tail.length > 0
          ? html`<pre tabindex="0">${job.log_tail.join("\n")}</pre>`
          : ""}

        <div class="help">
          Runs over <code>dataset/raw/</code>; violators are quarantined to
          <code>dataset/rejected/</code> with reasons. The linter above is the same gate, so a
          script that passes here will not be rejected there.
        </div>
      </section>
    `;
  }
}