import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  define: {
    'process.env.VITE_DEV_SERVER_URL': mode === 'development' ? '"http://localhost:5173"' : 'undefined'
  }
}));
