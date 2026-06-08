import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  root: path.resolve(__dirname),
  clearScreen: false,
  server: {
    port: parseInt(process.env.PORT || "1420", 10),
    strictPort: false,
    host: true,
    watch: {
      ignored: ["**/src-tauri/**", "**/python-sidecar/**"],
    },
  },
});
