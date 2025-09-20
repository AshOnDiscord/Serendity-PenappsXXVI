const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs')

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: true,
      contextIsolation: true,
    },
  });

  const startUrl =
    process.env.VITE_DEV_SERVER_URL ||
    `file://${path.join(__dirname, 'dist', 'index.html')}`;

  win.loadURL(startUrl);
}

app.on('ready', createWindow);

ipcMain.on('save-json', async (_event, data) => {
  const win = BrowserWindow.getFocusedWindow()
  if (!win) return

  const { filePath } = await dialog.showSaveDialog(win, {
    title: 'Save JSON',
    defaultPath: 'document.json',
    filters: [{ name: 'JSON', extensions: ['json'] }],
  })

  if (filePath) {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2))
  }
})


ipcMain.handle('open-json', async () => {
  const win = BrowserWindow.getFocusedWindow()
  if (!win) return null

  const { filePaths } = await dialog.showOpenDialog(win, {
    title: 'Open JSON',
    filters: [{ name: 'JSON', extensions: ['json'] }],
    properties: ['openFile'],
  })

  if (filePaths && filePaths[0]) {
    try {
      const content = fs.readFileSync(filePaths[0], 'utf-8')
      return JSON.parse(content)
    } catch (err) {
      console.error('Failed to read JSON:', err)
      return null
    }
  }

  return null
})



app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
