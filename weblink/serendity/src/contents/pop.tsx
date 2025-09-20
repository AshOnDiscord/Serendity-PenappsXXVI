import cssText from "data-text:~style.css"
import { Bookmark } from "lucide-react"
import type { PlasmoCSConfig } from "plasmo"
import { useReducer } from "react"

import { Glass } from "~glass"

export const config: PlasmoCSConfig = {
  matches: ["<all_urls>"],
  css: ["../style.css"]
}

export const getStyle = () => {
  const style = document.createElement("style")
  style.textContent = cssText
  return style
}

const PlasmoOverlay = () => {
  const color = document.body.style.backgroundColor
  const theme = isLightColor(color)
  const isDark = theme === "dark"

  return (
    <div
      className={`plasmo-font-sans plasmo-fixed plasmo-right-4 plasmo-top-4 plasmo-rounded-2xl`}>
      <Glass
        styler={
          isDark
            ? "plasmo-bg-[rgba(255,255,255,0.1)]"
            : "plasmo-bg-[rgba(0,0,0,0.1)]"
        }>
        <div className="plasmo-px-5 plasmo-py-[1.375rem]">
          <h1 className="plasmo-font-[Mortend-Bold] plasmo-text-xl">SEN_DEX</h1>
          <div className="plasmo-flex plasmo-flex-col plasmo-gap-4">
            <div className="plasmo-flex plasmo-gap-1.5 plasmo-h-10">
              <button className="plasmo-flex">
                <Glass
                  topStyle="plasmo-w-10 plasmo-flex plasmo-items-center plasmo-justify-center [--corner-radius:0.5rem!important] plasmo-border plasmo-border-white plasmo-rounded-[0.5rem]"
                  styler={
                    isDark
                      ? "plasmo-bg-[rgba(255,255,255,0.1)]"
                      : "plasmo-bg-[rgba(0,0,0,0.1)]"
                  }>
                  <Bookmark className="plasmo-w-4 plasmo-h-4" />
                </Glass>
              </button>
              <button className="plasmo-flex plasmo-font-[Mortend-Bold] plasmo-text-xs">
                <Glass
                  topStyle="plasmo-w-36 plasmo-flex plasmo-items-center plasmo-justify-center [--corner-radius:0.5rem!important] plasmo-border plasmo-border-white plasmo-rounded-[0.5rem]"
                  styler={
                    isDark
                      ? "plasmo-bg-[rgba(255,255,255,0.1)]"
                      : "plasmo-bg-[rgba(0,0,0,0.1)]"
                  }>
                  Research
                </Glass>
              </button>
            </div>
            <div className="plasmo-grid">
              <button className="plasmo-grid plasmo-h-10">
                <Glass
                  topStyle="plasmo-flex plasmo-items-center plasmo-justify-center [--corner-radius:0.5rem!important] plasmo-border plasmo-border-white  plasmo-rounded-[0.5rem]"
                  styler={
                    isDark
                      ? "plasmo-bg-[rgba(255,255,255,0.1)]"
                      : "plasmo-bg-[rgba(0,0,0,0.1)]"
                  }>
                  <span className="plasmo-text-orange-700 plasmo-text-xl">
                    +
                  </span>
                </Glass>
              </button>
            </div>
          </div>
        </div>
      </Glass>
    </div>
  )
}

export default PlasmoOverlay

function isLightColor(color) {
  // Convert color to RGB if it's in hex format
  let r, g, b
  if (color.startsWith("#")) {
    // Hex format (#RRGGBB or #RGB)
    if (color.length === 4) {
      r = parseInt(color[1] + color[1], 16)
      g = parseInt(color[2] + color[2], 16)
      b = parseInt(color[3] + color[3], 16)
    } else {
      r = parseInt(color[1] + color[2], 16)
      g = parseInt(color[3] + color[4], 16)
      b = parseInt(color[5] + color[6], 16)
    }
  } else if (color.startsWith("rgb")) {
    // RGB format (rgb(r, g, b))
    const rgb = color.match(/\d+/g)
    r = parseInt(rgb[0])
    g = parseInt(rgb[1])
    b = parseInt(rgb[2])
  } else {
    // Add support for other formats as needed
    return "light" // or handle error
  }

  // Apply luminance formula
  const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

  // Return whether the color is light or dark
  return luminance > 128 ? "light" : "dark"
}
