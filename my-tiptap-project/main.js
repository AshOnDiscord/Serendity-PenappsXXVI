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


app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
