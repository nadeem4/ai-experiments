import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// `@/...` is the same alias tsconfig.json gives the app, so tests import what the
// pages import -- including the exported data files, which is the point: the
// numbers on the page are asserted against the run they came from.
export default defineConfig({
  resolve: { alias: { "@": fileURLToPath(new URL(".", import.meta.url)) } },
});
