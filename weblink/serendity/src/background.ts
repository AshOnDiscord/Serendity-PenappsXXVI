chrome.action.onClicked.addListener(() => {
  manualSave()
})

chrome.commands.onCommand.addListener((command) => {
  if (command === "test") {
    manualSave(true)
  }
})

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  chrome.windows.get(tab.windowId, (win) => {
    if (!win.focused) return
    chrome.tabs.get(tabId, (tab) => {
      if (!tab.active) return
      if (changeInfo.status !== "complete") return
      console.log(
        `tab updated: ${tabId} - ${tab.windowId} ${changeInfo.status} ${tab.url}`
      )
    })
  })
})

chrome.tabs.onActivated.addListener((activeInfo) => {
  console.log(`tab activated: ${activeInfo.tabId} ${activeInfo.windowId}`)
})

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
