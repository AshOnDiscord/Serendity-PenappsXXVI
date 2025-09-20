export {}

declare global {
  interface Window {
    electronAPI: {
      saveJSON: (data: any) => void
      openJSON: () => Promise<any> // new
    }
  }
}