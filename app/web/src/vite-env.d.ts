/// <reference types="vite/client" />

/**
 * Vite exposes env vars through `import.meta.env`. The one setting this app
 * needs is the sidecar origin; declaring it here keeps `import.meta.env`
 * typed under `strict` instead of falling back to `any`.
 */
interface ImportMetaEnv {
  /** HTTP origin of the Python sidecar, e.g. "http://127.0.0.1:8765". */
  readonly VITE_SAMMONIC_API?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}