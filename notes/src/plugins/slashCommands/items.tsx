import { Editor } from "@tiptap/react";
import { CommandItem } from "./CommandsList";
import {
  Bold,
  Heading1,
  Heading2,
  Heading3,
  Italic,
  List,
  ListOrdered,
  Minus,
  Type,
} from "lucide-react";

interface CommandParams {
  editor: Editor;
  range: any;
}

const getSuggestionItems = ({ query }: { query: string }): CommandItem[] => {
  const items: CommandItem[] = [
    {
      title: "Text",
      icon: <Type className="w-4 h-4" />,
      command: ({ editor, range }: CommandParams) => {
        console.log("call some function from parent");
        editor.chain().focus().deleteRange(range).setNode("paragraph").run();
      },
    },
    {
      title: "H1",
      icon: <Heading1 className="w-4 h-4" />,
      command: ({ editor, range }: CommandParams) => {
        editor
          .chain()
          .focus()
          .deleteRange(range)
          .setNode("heading", { level: 1 })
          .run();
      },
    },
    {
      title: "H2",
      icon: <Heading2 className="w-4 h-4" />,
      command: ({ editor, range }: CommandParams) => {
        editor
          .chain()
          .focus()
          .deleteRange(range)
          .setNode("heading", { level: 2 })
          .run();
      },
    },
    {
      title: "H3",
      icon: <Heading3 className="w-4 h-4" />,
      command: ({ editor, range }: CommandParams) => {
        editor
          .chain()
          .focus()
          .deleteRange(range)
          .setNode("heading", { level: 3 })
          .run();
      },
    },
    {
      title: "bold",
      icon: <Bold className="w-4 h-4" />,
      command: ({ editor, range }: CommandParams) => {
        editor.chain().focus().deleteRange(range).setMark("bold").run();
      },
    },
    {
      title: "italic",
      icon: <Italic className="w-4 h-4" />,
      command: ({ editor, range }: CommandParams) => {
        editor.chain().focus().deleteRange(range).setMark("italic").run();
      },
    },
    {
      title: "bullet list",
      icon: <List className="w-4 h-4" />,
      command: ({ editor, range }: CommandParams) => {
        editor
          .chain()
          .focus()
          .deleteRange(range)
          .wrapInList("bulletList")
          .run();
      },
    },
    {
      title: "ordered list",
      icon: <ListOrdered className="w-4 h-4" />,
      command: ({ editor, range }: CommandParams) => {
        editor
          .chain()
          .focus()
          .deleteRange(range)
          .wrapInList("orderedList")
          .run();
      },
    },
    {
      title: "separator",
      icon: <Minus className="w-4 h-4" />,
      command: ({ editor, range }: CommandParams) => {
        editor.chain().focus().deleteRange(range).setHorizontalRule().run();
      },
    },
  ];

  return items
    .filter((item) => item.title.toLowerCase().startsWith(query.toLowerCase()))
    .slice(0, 10);
};

export default getSuggestionItems;
