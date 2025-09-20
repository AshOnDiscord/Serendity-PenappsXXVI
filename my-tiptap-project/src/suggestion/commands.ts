import { Extension } from "@tiptap/core";
import Suggestion, { SuggestionOptions } from "@tiptap/suggestion";
import { Editor } from "@tiptap/react";

interface CommandProps {
  editor: Editor;
  range: any;
  props: any;
}

declare module "@tiptap/core" {
  interface Commands<ReturnType> {
    mention: {
      /**
       * Insert a mention
       */
      insertMention: (options: { id: string; label?: string }) => ReturnType;
    };
  }
}

const Commands = Extension.create({
  name: "mention",

  addOptions() {
    return {
      suggestion: {
        char: "/",
        startOfLine: false,
        command: ({ editor, range, props }: CommandProps) => {
          props.command({ editor, range, props });
        }
      } as Partial<SuggestionOptions>
    };
  },

  addProseMirrorPlugins() {
    return [
      Suggestion({
        editor: this.editor,
        ...this.options.suggestion
      })
    ];
  }
});

export default Commands;