// global.d.ts
export {}

declare global {
  interface Window {
    electronAPI: {
      saveJSON: (data: any) => void
    }
  }
}
