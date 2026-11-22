import { defineConfig } from "vite";

/**
 * Vite config for the Sammonic Lit frontend.
 *
 * Two decisions worth naming:
 *
 * 1. `base: "./"` — the built bundle resolves assets relative to the document,
 *    so the same `dist/` works from a dev subdirectory, a static file server,
 *    or the future Tauri shell's custom protocol. No hardcoded origin.
 *
 * 2. No dev-server proxy. The browser talks straight to the sidecar origin
 *    (see src/api.ts). A proxy would be a second place to keep in sync when
 *    VITE_SAMMONIC_API changes, and the sidecar is loopback-only by design,
 *    so CORS is not a factor once it sends Access-Control headers — it does
 *    not, so the UI is served and used against the loopback origin directly.
 */
export default defineConfig({
  base: "./",
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    target: "es2022",
  },
});