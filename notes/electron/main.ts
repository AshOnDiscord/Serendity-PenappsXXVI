import { app, BrowserWindow } from "electron";
// import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
import express, { RequestHandler } from "express";
import cors from "cors";
import bodyParser from "body-parser";
import cleanURL from "./cleanUrl";

// const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// The built directory structure
//
// ├─┬─┬ dist
// │ │ └── index.html
// │ │
// │ ├─┬ dist-electron
// │ │ ├── main.js
// │ │ └── preload.mjs
// │
process.env.APP_ROOT = path.join(__dirname, "..");

// 🚧 Use ['ENV_NAME'] avoid vite:define plugin - Vite@2.x
export const VITE_DEV_SERVER_URL = process.env["VITE_DEV_SERVER_URL"];
export const MAIN_DIST = path.join(process.env.APP_ROOT, "dist-electron");
export const RENDERER_DIST = path.join(process.env.APP_ROOT, "dist");

process.env.VITE_PUBLIC = VITE_DEV_SERVER_URL
  ? path.join(process.env.APP_ROOT, "public")
  : RENDERER_DIST;

let win: BrowserWindow | null;

function createWindow() {
  win = new BrowserWindow({
    titleBarStyle: "hidden",
    icon: path.join(process.env.VITE_PUBLIC, "electron-vite.svg"),
    webPreferences: {
      preload: path.join(__dirname, "preload.mjs"),
    },
  });

  // Test active push message to Renderer-process.
  win.webContents.on("did-finish-load", () => {
    win?.webContents.send("main-process-message", new Date().toLocaleString());
  });

  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL);
  } else {
    // win.loadFile('dist/index.html')
    win.loadFile(path.join(RENDERER_DIST, "index.html"));
  }
}

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
    win = null;
  }
});

app.on("activate", () => {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.whenReady().then(createWindow);

const api = express();
api.use(
  cors({
    origin: "*", // or "*" for any origin (less secure)
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type"],
  })
);
api.use(bodyParser.json());

// api.get("/ping", (req, res) => {
//   res.json({ message: "pong" });
// });

interface ServerTabAPI {
  type: "UPDATE" | "REMOVE" | "FORCE";
  isActive?: boolean;
  tabId?: number;
  url?: string;
  timestamp: number;
}

let currentTab:
  | {
      tabId: number;
      url?: string;
      startTimestamp: number;
    }
  | undefined = undefined;

const tabs: {
  tabId: number;
  url?: string;
  duration: number;
}[] = [];

const history: {
  url: string;
  duration: number;
  timestamp: number;
}[] = [];

api.post("/tab-update", ((req, res) => {
  const url = req.body.url;
  const data = {
    ...req.body,
    url: url?.includes("//") ? cleanURL(url) : url,
  } as ServerTabAPI;
  if (data.type === "FORCE") {
    updateDBSite({
      tabId: data.tabId!,
      url: data.url,
      duration: 0,
    });
    return;
  } else if (data.type === "REMOVE") {
    // remove from tab list
    const index = tabs.findIndex((t) => t.tabId === data.tabId);
    let duration = 0;
    if (currentTab && currentTab.tabId === data.tabId) {
      duration = data.timestamp - currentTab.startTimestamp;
    }
    if (index !== -1) {
      tabs[index].duration += duration;
      updateDBSite(tabs[index]);
      tabs.splice(index, 1);
    }
  } else if (data.type === "UPDATE") {
    // check if we need to add a new tab
    const index = tabs.findIndex((t) => t.tabId === data.tabId);
    if (index === -1) {
      if (data.url) {
        tabs.push({
          tabId: data.tabId,
          url: data.url,
          duration: 0,
        });
      }
    } else {
      // check if we need to the url
      if (data.url && tabs[index].url !== data.url) {
        if (currentTab && currentTab.tabId === data.tabId) {
          tabs[index].duration += data.timestamp - currentTab.startTimestamp;
        }
        updateDBSite(tabs[index]);
        tabs[index].url = data.url;
        tabs[index].duration = 0; // reset duration for new url
      }
    }
    // check if we need to update current tab
    if (data.isActive) {
      if (currentTab && currentTab.tabId !== data.tabId) {
        // update previous active's duration
        const prevIndex = tabs.findIndex((t) => t.tabId === currentTab!.tabId);
        if (prevIndex !== -1) {
          tabs[prevIndex].duration +=
            data.timestamp - currentTab.startTimestamp;
        }
      }
      currentTab = {
        tabId: data.tabId,
        url: data.url,
        startTimestamp: data.timestamp,
      };
    }
  }
  res.json({ message: "ok" });
}) as RequestHandler<object, object, ServerTabAPI>);

api.get("/get-recent", (req, res) => {
  res.json({ history });
});

const updateDBSite = ({ url, duration }: (typeof tabs)[0]) => {
  if (!url) return;
  console.log("LINK CLOSED", { url, duration, timestamp: Date.now() });
  history.push({ url, duration, timestamp: Date.now() });
};

api.listen(4000, () => {
  console.log("API server running at http://localhost:4000");
});
