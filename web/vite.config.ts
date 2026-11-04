import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI backend (agentto.app.server) listens on 127.0.0.1:8765.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
    },
  },
});
