import { ReactRenderer } from "@tiptap/react";
import tippy, { Instance as TippyInstance } from "tippy.js";
import CommandsList, { CommandItem } from "./CommandsList";

interface RenderItemsProps {
  items: CommandItem[];
  command: (item: CommandItem) => void;
  editor: any;
  clientRect: () => DOMRect;
}

interface KeyDownProps {
  event: KeyboardEvent;
}

const renderItems = () => {
  let component: ReactRenderer;
  let popup: TippyInstance[];

  return {
    onStart: (props: RenderItemsProps) => {
      component = new ReactRenderer(CommandsList, {
        props,
        editor: props.editor
      });

      popup = tippy("body", {
        getReferenceClientRect: props.clientRect,
        appendTo: () => document.body,
        content: component.element,
        showOnCreate: true,
        interactive: true,
        trigger: "manual",
        placement: "bottom-start"
      });
    },
    onUpdate(props: RenderItemsProps) {
      component.updateProps(props);

      popup[0].setProps({
        getReferenceClientRect: props.clientRect
      });
    },
    onKeyDown(props: KeyDownProps): boolean {
      if (props.event.key === "Escape") {
        popup[0].hide();
        return true;
      }

      // Type assertion to access the onKeyDown method
      const commandsListRef = component.ref as CommandsList | null;
      return commandsListRef?.onKeyDown?.(props) || false;
    },
    onExit() {
      popup[0].destroy();
      component.destroy();
    }
  };
};

export default renderItems;