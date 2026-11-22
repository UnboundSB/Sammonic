/**
 * hypno-app — the application shell.
 *
 * Responsibilities:
 *  - render the header: API origin, GPU busy state, tracked job count, notices;
 *  - own the two background polls (GET /jobs for GPU state, GET /status/:id for
 *    the one watched job) so two panels can never double-poll the same job, and
 *    a job started by a handoff still renders in the right panel;
 *  - clear a panel's busy flag when its job reaches a terminal state.
 *
 * Polling lives here rather than in the panels on purpose: the sidecar is
 * single-threaded per session for edits and serializes GPU work for jobs, so
 * there is never a reason for more than one status poll in flight.
 */

import { LitElement, html, css } from "lit";
import { customElement, state } from "lit/decorators.js";

import "./script-editor";
import "./lint-panel";
import "./dataset-panel";
import "./train-panel";
import "./generate-panel";

import * as api from "./api";
import { store, panelForKind } from "./state";
import type { AppState } from "./state";

const JOBS_POLL_MS = 2000;
const STATUS_POLL_MS = 1000;

@customElement("hypno-app")
export class HypnoApp extends LitElement {
  static override styles = css`
    :host {
      /* Theme tokens. Panels read these through the shadow-DOM cascade,
         so one palette definition serves every panel. */
      --bg: #14161a;
      --panel-bg: #1b1e24;
      --panel-border: #2c313a;
      --fg: #e6e8ec;
      --fg-dim: #9aa3af;
      --accent: #7cc7ff;
      --ok: #6fd08c;
      --warn: #e3b341;
      --error: #ef6a6a;
      display: block;
      background: var(--bg);
      color: var(--fg);
      min-height: 100vh;
      font-family: system-ui, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      gap: 8px 18px;
      padding: 10px 16px;
      border-bottom: 1px solid var(--panel-border);
      background: #171a1f;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    h1 {
      font-size: 15px;
      margin: 0;
      letter-spacing: 0.04em;
    }
    .origin { color: var(--fg-dim); font-family: ui-monospace, monospace; font-size: 12px; }
    .gpu { font-size: 12px; padding: 2px 8px; border-radius: 10px; border: 1px solid var(--panel-border); }
    .gpu.idle { color: var(--ok); }
    .gpu.busy { color: var(--warn); border-color: var(--warn); }
    .count { color: var(--fg-dim); font-size: 12px; }
    .notice { margin-left: auto; font-size: 12px; max-width: 60ch; }
    .notice.info { color: var(--accent); }
    .notice.error { color: var(--error); }
    .notice button {
      background: none; border: none; color: inherit; cursor: pointer;
      font-size: 12px; padding: 0 4px;
    }
    main {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 12px;
      padding: 12px 16px 24px;
      align-items: start;
    }
    script-editor { grid-column: 1 / -1; }
  `;

  @state() private app: AppState = store.getState();

  private unsubscribe: (() => void) | null = null;
  private jobsTimer: number | null = null;
  private statusTimer: number | null = null;

  override connectedCallback(): void {
    super.connectedCallback();
    this.unsubscribe = store.subscribe((next) => {
      this.app = next;
    });
    void this.pollJobs();
    void this.pollWatched();
    this.jobsTimer = window.setInterval(() => void this.pollJobs(), JOBS_POLL_MS);
    this.statusTimer = window.setInterval(() => void this.pollWatched(), STATUS_POLL_MS);
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    if (this.jobsTimer !== null) window.clearInterval(this.jobsTimer);
    if (this.statusTimer !== null) window.clearInterval(this.statusTimer);
    this.jobsTimer = null;
    this.statusTimer = null;
    this.unsubscribe?.();
    this.unsubscribe = null;
  }

  /** GET /jobs — GPU occupancy and every job the sidecar still remembers. */
  private async pollJobs(): Promise<void> {
    try {
      const response = await api.jobs();
      store.setState({
        jobs: response.jobs,
        gpuBusyWith: response.gpu_busy_with,
      });
    } catch (err) {
      store.setNotice({ level: "error", text: api.errorMessage(err) });
    }
  }

  /**
   * GET /status/:id for the watched job. On a terminal state the watch stops
   * (the snapshot stays in `store.job` for the panel to render) and the
   * acting panel's busy flag is cleared here, not in the panel, so the flag
   * cannot get stuck if the panel unmounts mid-job.
   */
  private async pollWatched(): Promise<void> {
    const { watched, job } = this.app;
    if (watched === null) return;
    if (job !== null && api.isTerminalJobState(job.state)) {
      store.stopWatching();
      return;
    }
    try {
      const status = await api.status(watched.id);
      if (this.app.watched?.id !== watched.id) return; // superseded
      store.setJob(status);
      if (api.isTerminalJobState(status.state)) {
        const panel = panelForKind(status.kind);
        if (panel !== null) store.setBusy(panel, false);
        store.stopWatching();
      }
    } catch (err) {
      const panel = panelForKind(watched.kind);
      if (panel !== null) store.setBusy(panel, false);
      store.stopWatching();
      store.setNotice({ level: "error", text: api.errorMessage(err) });
    }
  }

  private dismissNotice(): void {
    store.setNotice(null);
  }

  override render() {
    const { apiOrigin, gpuBusyWith, jobs, notice } = this.app;
    const running = jobs.filter((j) => j.state === "running" || j.state === "queued").length;

    return html`
      <header>
        <h1>SAMMONIC</h1>
        <span class="origin" title="Every panel talks to this origin only">api: ${apiOrigin}</span>
        <span class="gpu ${gpuBusyWith === null ? "idle" : "busy"}" title="Training and TTS share the GPU and never run at the same time">
          gpu: ${gpuBusyWith === null ? "idle" : `busy — ${gpuBusyWith}`}
        </span>
        <span class="count">${jobs.length} job${jobs.length === 1 ? "" : "s"} tracked, ${running} active</span>
        ${notice === null
          ? ""
          : html`<span class="notice ${notice.level}" role="status">
              ${notice.text}
              <button type="button" @click=${this.dismissNotice} aria-label="dismiss">✕</button>
            </span>`}
      </header>
      <main>
        <script-editor></script-editor>
        <lint-panel></lint-panel>
        <dataset-panel></dataset-panel>
        <train-panel></train-panel>
        <generate-panel></generate-panel>
      </main>
    `;
  }
}