chrome.action.onClicked.addListener(() => {
  manualSave()
})

chrome.commands.onCommand.addListener((command) => {
  if (command === "test") {
    manualSave(true)
  }
})

interface ServerTabAPI {
  type: "UPDATE" | "REMOVE" | "FORCE"
  isActive?: boolean
  tabId: number
  url?: string
  timestamp: number
}

chrome.tabs.onCreated.addListener((tab) => {
  console.log(`tab created: ${tab.id} ${tab.url}`)
  chrome.windows.get(tab.windowId, (window) => {
    const data: ServerTabAPI = {
      type: "UPDATE",
      isActive: window.focused && tab.active,
      tabId: tab.id!,
      url: tab.url,
      timestamp: Date.now()
    }
    sendTabData(data)
  })
})

chrome.tabs.onRemoved.addListener((tabId, removeInfo) => {
  console.log(`tab removed: ${tabId}`)
  const data: ServerTabAPI = {
    type: "REMOVE",
    isActive: undefined,
    tabId: tabId,
    url: undefined,
    timestamp: Date.now()
  }
  sendTabData(data)
})

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete") return
  chrome.windows.get(tab.windowId, (window) => {
    const data: ServerTabAPI = {
      type: "UPDATE",
      isActive: window.focused && tab.active,
      tabId: tab.id!,
      url: tab.url,
      timestamp: Date.now()
    }
    sendTabData(data)
  })
})

chrome.tabs.onActivated.addListener((activeInfo) => {
  chrome.tabs.get(activeInfo.tabId, (tab) => {
    const data: ServerTabAPI = {
      type: "UPDATE",
      isActive: tab.active,
      tabId: tab.id!,
      url: tab.url,
      timestamp: Date.now()
    }
    sendTabData(data)
  })
})

const sendTabData = (data: ServerTabAPI) => {
  fetch("http://localhost:4000/tab-update", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(data as ServerTabAPI)
  })
    .then((res) => {
      if (!res.ok) {
        console.error("Failed to send tab data", res.statusText)
      }
    })
    .catch((err) => {
      console.error("Error sending tab data", err)
    })
}

const manualSave = (keybindTriggered = false) => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs[0]
    if (tab) {
      console.log(
        `${keybindTriggered ? "keybind" : "button"} pressed ${tab.id} - ${tab.windowId} ${tab.url}`
      )
      chrome.tabs.sendMessage(tab.id, { type: "GET_VIEWPORT" }, (res) => {
        if (!res) {
          console.error("No response from content script")
          return
        }
        console.log("Viewport", res)
      })
    }
  })
}
