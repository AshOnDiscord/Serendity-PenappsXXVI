import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// https://vitejs.dev/config/
export default defineConfig({
  base: "",
  plugins: [svelte()],
  worker: { format: "es" },
  build: {
    target: "esnext",
    chunkSizeWarningLimit: 4096,
  },
});
