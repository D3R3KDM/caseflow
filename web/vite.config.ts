import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// web app on 5190; /api goes to the local CaseFlow engine on 8600
export default defineConfig({
  plugins: [react()],
  server: { port: 5190, strictPort: true, proxy: { '/api': 'http://127.0.0.1:8600' } },
})
