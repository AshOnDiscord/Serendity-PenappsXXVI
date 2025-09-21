<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { createElement } from "react";
  import { createRoot } from "react-dom/client";

  let { component, props } = $props();

  function mountReactComponent(element: HTMLDivElement, props: { component: any; props: any }) {
    const component = props.component;

    let root = createRoot(element);
    root.render(createElement(component, { props: props.props }));
    return {
      destroy() {
        root.unmount();
      },
    };
  }
</script>

<div use:mountReactComponent={{ component: component, props: props }} class="w-full h-full"></div>
