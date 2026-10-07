import { defineConfig } from "vite";

const engine = process.env.AIOTECH_ENGINE_URL ?? "http://127.0.0.1:8000";
const webKey = process.env.AIOTECH_WEB_API_KEY ?? "";

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/api/ui": {
        target: engine,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/ui/, "/v1"),
        headers: webKey ? { "x-api-key": webKey } : {},
      },
    },
  },
  build: {
    target: "es2022",
    sourcemap: false,
    chunkSizeWarningLimit: 900,
  },
});
