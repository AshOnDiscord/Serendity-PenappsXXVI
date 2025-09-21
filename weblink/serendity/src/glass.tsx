export const Glass = ({ children, styler, topStyle = "" }) => {
  return (
    <div className={`GlassContainer ${topStyle}`}>
      <div className="GlassContent">{children}</div>
      <div
        className={`GlassMaterial ${styler
          .split(" ")
          .map((c) => `after:${c}`)
          .join(" ")}`}>
        <div className="GlassEdgeReflection"></div>
        <div className="GlassEmbossReflection"></div>
        <div className="GlassRefraction"></div>
        <div className="GlassBlur"></div>
        <div className="BlendLayers"></div>
        <div className="BlendEdge"></div>
        <div className="Highlight"></div>
        <div className="Tint"></div>
      </div>
    </div>
  )
}
