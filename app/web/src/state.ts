/**
 * The Sammonic store: one small mutable object with a subscribe/emit API
 * sized for Lit components. No Redux, no signals, no framework.
 *
 * How Lit uses it: a component calls `store.subscribe(...)` in
 * `connectedCallback` and stashes the snapshot into a `@state()` field, so
 * every store change schedules exactly one render of that component. The
 * store replaces its state object on every mutation (never mutates in place),
 * which keeps `changedProperties` semantics trivial and snapshots stable for
 * the lifetime of a render.
 *
 * What lives here (per architecture.md §3.1): the current script text, the
 * lint result, job status (one watched job at a time), the job's log tail
 * (carried inside `job.log_tail`), per-panel busy flags, and the GPU state
 * the header displays. Panels never talk to each other directly — a handoff
 * from the editor to the generate panel goes through this store.
 */

import * as api from "./api";
import type {
  JobState,
  JobStatus,
  JobSummary,
  LintResult,
} from "./api";

/** Busy flags are per panel, not global: only the acting panel blocks. */
export type PanelName = "editor" | "lint" | "dataset" | "train" | "generate";

export interface BusyFlags {
  editor: boolean;
  lint: boolean;
  dataset: boolean;
  train: boolean;
  generate: boolean;
}

/** A script handed from the editor to the generate panel. */
export interface ScriptHandoff {
  /** Repo-relative script path the sidecar will read (see note in script-editor). */
  path: string;
  /** The handed-over text, for display and for the lint panel's baseline. */
  text: string;
  /** UTF-8 byte length of `text`, as the sidecar counts it. */
  bytes: number;
}

export interface AppNotice {
  level: "info" | "error";
  text: string;
}

export interface WatchedJob {
  kind: string;
  id: string;
}

export interface AppState {
  /** Origin the UI is talking to (mirrors api.API_ORIGIN for display). */
  apiOrigin: string;
  /** Mirror of the editor's document — the piece table stays the source of truth. */
  scriptText: string;
  /** Most recent /lint response, or null before the first successful run. */
  lint: LintResult | null;
  /** The job the shell is polling, if any. */
  watched: WatchedJob | null;
  /** Latest snapshot of the watched job (its log_tail is the job log tail). */
  job: JobStatus | null;
  /** Who holds the GPU right now (from GET /jobs). */
  gpuBusyWith: string | null;
  /** Every job the sidecar still tracks (from GET /jobs). */
  jobs: JobSummary[];
  /** Paths the generate panel edits; the editor handoff reads them too. */
  scriptPath: string;
  referencePath: string;
  /** Set when the editor hands its document to the generate panel. */
  handoff: ScriptHandoff | null;
  busy: BusyFlags;
  notice: AppNotice | null;
}

/** Job kinds that map 1:1 onto a panel busy flag. */
const KIND_TO_PANEL: Record<string, PanelName> = {
  dataset: "dataset",
  train: "train",
  generate: "generate",
};

export function panelForKind(kind: string): PanelName | null {
  return KIND_TO_PANEL[kind] ?? null;
}

function initialState(): AppState {
  return {
    apiOrigin: api.API_ORIGIN,
    scriptText: "",
    lint: null,
    watched: null,
    job: null,
    gpuBusyWith: null,
    jobs: [],
    scriptPath: "script.txt",
    referencePath: "reference_audio.wav",
    handoff: null,
    busy: {
      editor: true, // the editor opens its piece-table session on connect
      lint: false,
      dataset: false,
      train: false,
      generate: false,
    },
    notice: null,
  };
}

type Listener = (state: AppState) => void;

let state: AppState = initialState();
const listeners = new Set<Listener>();

function emit(): void {
  // Copy the set first: a listener may unsubscribe during notification.
  for (const listener of [...listeners]) listener(state);
}

function update(patch: Partial<AppState>): void {
  state = { ...state, ...patch };
  emit();
}

export const store = {
  getState(): AppState {
    return state;
  },

  subscribe(listener: Listener): () => void {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },

  setState(patch: Partial<AppState>): void {
    update(patch);
  },

  setBusy(panel: PanelName, value: boolean): void {
    if (state.busy[panel] === value) return;
    update({ busy: { ...state.busy, [panel]: value } });
  },

  setScriptText(text: string): void {
    if (state.scriptText === text) return;
    update({ scriptText: text });
  },

  setLint(lint: LintResult | null): void {
    update({ lint });
  },

  setScriptPath(path: string): void {
    if (state.scriptPath === path) return;
    update({ scriptPath: path });
  },

  setReferencePath(path: string): void {
    if (state.referencePath === path) return;
    update({ referencePath: path });
  },

  setHandoff(handoff: ScriptHandoff | null): void {
    update({ handoff });
  },

  /**
   * Notices are transient one-liners in the header. Identical consecutive
   * notices are dropped so a failing /jobs poll cannot re-render the app at
   * the poll rate forever.
   */
  setNotice(notice: AppNotice | null): void {
    const current = state.notice;
    if (
      current === notice ||
      (current !== null &&
        notice !== null &&
        current.level === notice.level &&
        current.text === notice.text)
    ) {
      return;
    }
    update({ notice });
  },

  /** Starts watching a job: clears the previous snapshot and its busy flag. */
  startWatching(kind: string, jobId: string): void {
    update({ watched: { kind, id: jobId }, job: null });
  },

  setJob(job: JobStatus): void {
    // A snapshot always replaces the previous one; the sidecar sends the
    // whole status (including the log tail) on every poll, so merging would
    // only ever duplicate fields.
    update({ job });
  },

  /** Stops polling but keeps the last snapshot on screen. */
  stopWatching(): void {
    update({ watched: null });
  },

  /**
   * The most recent job snapshot for a panel's kind, or null.
   *
   * Deliberately does not require `watched`: a finished job stays on screen
   * after the shell stops polling it — the dataset counts and the final MP3
   * duration must not vanish the instant the job completes. `job` is cleared
   * whenever a new watch starts, so a snapshot is always the latest job's.
   */
  watchedJob(kind: string): JobStatus | null {
    const { job } = state;
    if (job === null || job.kind !== kind) return null;
    return job;
  },
};

export type { JobState };