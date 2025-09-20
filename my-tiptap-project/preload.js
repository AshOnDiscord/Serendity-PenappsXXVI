const {ipcRenderer} = require('electron/renderer')
const {contextBridge} = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  saveJSON: (data) => ipcRenderer.send('save-json', data),
  openJSON: () => ipcRenderer.invoke('open-json')
})
