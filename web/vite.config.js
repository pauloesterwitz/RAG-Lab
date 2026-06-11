import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// In dev, proxy /api to the FastAPI backend on :8000.
// In prod, `vite build` emits to dist/, which FastAPI serves directly.
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
