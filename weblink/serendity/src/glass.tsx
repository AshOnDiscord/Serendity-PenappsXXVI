export const Glass = ({ children, styler }) => {
  return (
    <div className="GlassContainer">
      <div className="GlassContent">{children}</div>
      <div className={`GlassMaterial after:${styler}`}>
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
