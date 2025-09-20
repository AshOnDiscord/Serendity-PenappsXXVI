interface Viewport {
  start: number
  end: number
}

interface ElementData {
  topVisible: boolean
  bottomVisible: boolean
  text: string | null
}

const DEBUG_OUTLINE = false

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type !== "GET_VIEWPORT") return
  console.log("Received GET_VIEWPORT message from", sender)

  // grab contents in viewport
  const viewportStart = window.scrollY
  const viewportEnd = viewportStart + window.innerHeight
  const viewport: Viewport = { start: viewportStart, end: viewportEnd }

  // recurse downwards, filtering out text, maybe find a way to split text off subchild visiblity
  const data = visiblityFilter(document.body, viewport)
  sendResponse({ data: data.text })
  return true // keep channel open for async response
})

const visiblityFilter = (el: Element, viewport: Viewport): ElementData => {
  if (!isRenderable(el)) {
    return {
      topVisible: false,
      bottomVisible: false,
      text: null
    }
  }
  const rect = el.getBoundingClientRect()
  const elStart = rect.top + window.scrollY
  const elEnd = rect.bottom + window.scrollY
  // is any part of the element visible?
  const visible = elStart <= viewport.end && viewport.start <= elEnd
  if (!visible) {
    if (DEBUG_OUTLINE) (el as HTMLElement).style.outline = "2px solid red"
    return {
      topVisible: false,
      bottomVisible: false,
      text: null
    }
  }
  // set green outline
  if (DEBUG_OUTLINE) (el as HTMLElement).style.outline = "2px solid green"

  const children = el.childNodes
  if (children.length === 0) {
    return {
      topVisible: true,
      bottomVisible: true,
      text: ""
    }
  }
  if (children.length === 1 && children[0].nodeType === Node.TEXT_NODE) {
    return {
      topVisible: true,
      bottomVisible: true,
      text: (children[0] as Text).textContent.trim()
    }
  }

  const text = [] // just childNode w/ proper text, pruning is afterwards

  let start = undefined
  let end = undefined

  for (const [i, child] of children.entries()) {
    if (child.nodeType === Node.TEXT_NODE) {
      text.push(child.textContent.trim())
      continue
    }
    if (child.nodeType !== Node.ELEMENT_NODE) continue // ignore non-elements (including text, we'll come back)

    const childEl = child as Element
    const visibility = visiblityFilter(childEl, viewport)
    if (start === undefined && !visibility.topVisible) start = i
    if (end === undefined && !visibility.bottomVisible) end = i
    if (visibility.text) text.push(visibility.text.trim())
  }

  const prunedText = text.slice(start, end ? end + 1 : undefined).join(" ")
  return {
    // based off children
    topVisible: start === undefined,
    bottomVisible: end === undefined,
    text: prunedText.length > 0 ? prunedText : ""
  }
}

const isRenderable = (el: Element) => {
  const style = window.getComputedStyle(el)
  return (
    style.display !== "none" &&
    style.visibility !== "hidden" &&
    el.tagName !== "SCRIPT" &&
    el.tagName !== "STYLE"
  )
}
