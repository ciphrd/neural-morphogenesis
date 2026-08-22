import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// server.fs.allow: this project's own GPU physics shaders are NOT copied
// into src/ — they're loaded straight out of ../core/*.wgsl via `?raw`
// imports (see src/gpu/mpmCore.ts), the same single-source-of-truth
// convention trainer/shader_template.py's load_core_shader() already
// uses on the Python side. That path lives outside this package's own
// root, which Vite's dev server refuses to serve by default; "one level
// up" (mpm-training/) covers both viewer/ and its sibling core/.
export default defineConfig({
  plugins: [react()],
  server: {
    fs: {
      allow: [".."],
    },
  },
});
