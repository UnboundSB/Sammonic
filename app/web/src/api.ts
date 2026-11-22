/**
 * Sammonic sidecar HTTP client — the frontend's only backend dependency.
 *
 * Transport: ONE HTTP origin, the Python sidecar (default
 * http://127.0.0.1:8765, overridable at build/dev time with VITE_SAMMONIC_API).
 * There is deliberately no Node/koffi bridge in front of the sidecar: the
 * sidecar owns both the job routes (/lint, /dataset/build, /train, /generate,
 * /status, /job/<id>/cancel, /jobs) and the editing core (/edit/*), so the UI
 * has exactly one origin and one contract to match.
 *
 * Error contract: every non-2xx response carries a JSON body with an `error`
 * string (a 409 adds `running_job`). `request()` parses that body and rethrows
 * it as an ApiError carrying the status, the server's message, and the full
 * payload, so callers can say "GPU busy with <job>" instead of "request failed".
 *
 * Type policy: no `any` anywhere in this module. Responses are typed through
 * the exported interfaces; JSON arrives as `unknown` and is narrowed once,
 * here, instead of leaking untyped values into component props.
 */

/** Origin every request is sent to. */
export const API_ORIGIN: string =
  import.meta.env.VITE_SAMMONIC_API ?? "http://127.0.0.1:8765";

/** Job states the sidecar reports for train / generate / dataset jobs. */
export type JobState = "queued" | "running" | "done" | "error" | "cancelled";

/** States after which a job will never change again. */
export const TERMINAL_JOB_STATES: ReadonlySet<JobState> = new Set<JobState>([
  "done",
  "error",
  "cancelled",
]);

export function isTerminalJobState(state: JobState): boolean {
  return TERMINAL_JOB_STATES.has(state);
}

/** A failure that came back from the sidecar (or from not reaching it). */
export class ApiError extends Error {
  readonly status: number;
  readonly payload: Record<string, unknown>;

  constructor(
    status: number,
    message: string,
    payload: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

/** Human-readable message for anything thrown by this module. */
export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return String(err);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * One fetch to the sidecar. `method` is POST whenever a body is supplied;
 * every route in this client is either POST-with-JSON or GET.
 */
async function request<T>(path: string, body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_ORIGIN}${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (cause) {
    // Network-level failure (sidecar down, wrong origin, CORS). The browser
    // hides the status here, so name the origin we actually tried.
    throw new ApiError(0, `cannot reach sidecar at ${API_ORIGIN}: ${errorMessage(cause)}`);
  }

  const raw = await response.text();
  let payload: unknown = null;
  if (raw.trim().length > 0) {
    try {
      payload = JSON.parse(raw) as unknown;
    } catch {
      payload = null; // non-JSON body: fall through to the status-line message
    }
  }

  if (!response.ok) {
    const message =
      isRecord(payload) && typeof payload.error === "string"
        ? payload.error
        : `${response.status} ${response.statusText || "request failed"}`;
    throw new ApiError(response.status, message, isRecord(payload) ? payload : {});
  }
  // The sidecar always answers JSON; the cast is the single narrowing point.
  return payload as T;
}

// ------------------------------------------------------------------ lint

export interface LintResult {
  ok: boolean;
  errors: string[];
  warnings: string[];
  info: string[];
}

/**
 * POST /lint — run the format gate. The linter runs in-process on the
 * sidecar (pure stdlib, no GPU), so this is safe to call on every keystroke
 * burst via the lint panel's debounce.
 */
export function lint(text: string): Promise<LintResult> {
  return request<LintResult>("/lint", { text });
}

// ------------------------------------------------------------------ jobs

/** Payload shape of the last `@@PROGRESS {...}` heartbeat, keyed by job kind. */
export interface JobProgress {
  /** Chunk/script index (dataset build, generate). */
  i?: number;
  /** Total chunks/scripts. */
  total?: number;
  /** Training step. */
  step?: number;
  /** Total training steps. */
  total_steps?: number;
  /** Training loss, when the trainer emits it. */
  loss?: number;
  /** Current raw script name (dataset build). */
  name?: string;
  /** Chunk length in characters (generate). */
  chars?: number;
  /** Seconds spent on the current chunk (generate). */
  secs?: number;
}

/** `@@RESULT {...}` of a finished dataset build. */
export interface DatasetResult {
  scripts: number;
  accepted: number;
  rejected: number;
  samples: number;
  train: number;
  val: number;
}

/** `@@RESULT {...}` of a finished generate job. */
export interface GenerateResult {
  mp3: string;
  duration: number;
}

/**
 * `@@RESULT {...}` of a finished job. Train results are not yet shaped by any
 * producer (the trainer does not exist yet), so the known keys are optional and
 * unknown keys are simply not typed rather than forced through `any`.
 */
export type JobResult = DatasetResult | GenerateResult;

export interface JobStatus {
  id: string;
  kind: string;
  state: JobState;
  progress: JobProgress | null;
  returncode: number | null;
  error: string | null;
  elapsed_s: number | null;
  log_tail: string[];
  result: JobResult | null;
  meta: Record<string, string>;
  /** 0..1, computed by the sidecar from i/total or step/total_steps. */
  progress_fraction: number | null;
}

export interface DatasetBuildStart {
  job: string;
  raw_dir: string;
}

export interface TrainStart {
  job: string;
  config: string;
}

export interface GenerateStart {
  job: string;
  script: string;
  reference: string;
}

export interface JobSummary {
  id: string;
  kind: string;
  state: JobState;
  progress_fraction: number | null;
  error: string | null;
}

export interface JobsResponse {
  jobs: JobSummary[];
  /** Job id currently holding the GPU, or null when the card is free. */
  gpu_busy_with: string | null;
}

export interface CancelResult {
  id: string;
  cancelled: boolean;
  state: JobState;
  detail?: string;
}

/** POST /dataset/build — starts the CPU-side dataset job (202 Accepted). */
export function datasetBuild(rawDir?: string): Promise<DatasetBuildStart> {
  const body: { raw_dir?: string } = {};
  if (rawDir && rawDir.trim().length > 0) body.raw_dir = rawDir.trim();
  return request<DatasetBuildStart>("/dataset/build", body);
}

/** POST /train — starts a GPU job; 409s when the card is already taken. */
export function train(config?: string, resume?: string): Promise<TrainStart> {
  const body: { config?: string; resume?: string } = {};
  if (config && config.trim().length > 0) body.config = config.trim();
  if (resume && resume.trim().length > 0) body.resume = resume.trim();
  return request<TrainStart>("/train", body);
}

/** POST /generate — starts a TTS render job; 409s when the GPU is taken. */
export function generate(
  script?: string,
  reference?: string,
): Promise<GenerateStart> {
  const body: { script?: string; reference?: string } = {};
  if (script && script.trim().length > 0) body.script = script.trim();
  if (reference && reference.trim().length > 0) body.reference = reference.trim();
  return request<GenerateStart>("/generate", body);
}

/** GET /status/:id — one snapshot of a job. */
export function status(jobId: string): Promise<JobStatus> {
  return request<JobStatus>(`/status/${encodeURIComponent(jobId)}`);
}

/** POST /job/:id/cancel — asks a running job to stop. */
export function cancel(jobId: string): Promise<CancelResult> {
  return request<CancelResult>(`/job/${encodeURIComponent(jobId)}/cancel`, {});
}

/** GET /jobs — every tracked job plus who holds the GPU. */
export function jobs(): Promise<JobsResponse> {
  return request<JobsResponse>("/jobs");
}

// ------------------------------------------------------------ editing core
//
// The /edit/* routes are backed by the C++ piece-table DLL via Python ctypes.
// ALL positions are BYTE offsets into the UTF-8 logical document — never
// character indices. The editor (script-editor.ts) is the only caller and is
// responsible for both the conversion and for serializing its requests: the
// sidecar serves one session single-threaded, so two in-flight edits for the
// same session whose offsets were computed against different document states
// would corrupt the buffer.

/** POST /edit/open — opens a session seeded with `text`. */
export interface EditOpenResult {
  session: string;
  length: number;
  pieces: number;
}

export function editOpen(text: string): Promise<EditOpenResult> {
  return request<EditOpenResult>("/edit/open", { text });
}

/** Shared response of /edit/insert and /edit/erase. */
export interface EditOpResult {
  length: number;
  pieces: number;
  /** Sidecar's `pt_check_invariant` (piece lengths sum to the document length). */
  consistent?: boolean;
}

export function editInsert(
  session: string,
  pos: number,
  text: string,
): Promise<EditOpResult> {
  return request<EditOpResult>("/edit/insert", { session, pos, text });
}

export function editErase(
  session: string,
  start: number,
  end: number,
): Promise<EditOpResult> {
  return request<EditOpResult>("/edit/erase", { session, start, end });
}

/** POST /edit/undo — returns the full document text after the undo. */
export interface EditUndoResult {
  undone: boolean;
  length: number;
  text: string;
}

/** POST /edit/redo — returns the full document text after the redo. */
export interface EditRedoResult {
  redone: boolean;
  length: number;
  text: string;
}

export function editUndo(session: string): Promise<EditUndoResult> {
  return request<EditUndoResult>("/edit/undo", { session });
}

export function editRedo(session: string): Promise<EditRedoResult> {
  return request<EditRedoResult>("/edit/redo", { session });
}

/** GET-equivalent POST /edit/text — authoritative document text. */
export interface EditTextResult {
  text: string;
  length: number;
  pieces: number;
  consistent?: boolean;
}

export function editText(session: string): Promise<EditTextResult> {
  return request<EditTextResult>("/edit/text", { session });
}

/** POST /edit/close — releases a server-side session. */
export interface EditCloseResult {
  closed: boolean;
}

export function editClose(session: string): Promise<EditCloseResult> {
  return request<EditCloseResult>("/edit/close", { session });
}