import { Editor } from "@tiptap/react";
import { CommandItem } from "./CommandsList";

interface CommandParams {
  editor: Editor;
  range: any;
}

const getSuggestionItems = (query: string | { query: string } | any): CommandItem[] => {
  // Handle different query formats that might be passed
  let searchQuery = "";
  
  if (typeof query === "string") {
    searchQuery = query;
  } else if (query && typeof query.query === "string") {
    searchQuery = query.query;
  } else if (query && typeof query === "object") {
    // Sometimes the query might be an object with other properties
    searchQuery = query.text || query.search || "";
  }

  const items: CommandItem[] = [
    {
      title: "H1",
      command: ({ editor, range }: CommandParams) => {
        editor
          .chain()
          .focus()
          .deleteRange(range)
          .setNode("heading", { level: 1 })
          .run();
      }
    },
    {
      title: "H2",
      command: ({ editor, range }: CommandParams) => {
        editor
          .chain()
          .focus()
          .deleteRange(range)
          .setNode("heading", { level: 2 })
          .run();
      }
    },
    {
      title: "bold",
      command: ({ editor, range }: CommandParams) => {
        editor.chain().focus().deleteRange(range).setMark("bold").run();
      }
    },
    {
      title: "italic",
      command: ({ editor, range }: CommandParams) => {
        editor.chain().focus().deleteRange(range).setMark("italic").run();
      }
    },
    {
      title: "image",
      command: ({ editor, range }: CommandParams) => {
        console.log("call some function from parent");
        editor.chain().focus().deleteRange(range).setNode("paragraph").run();
      }
    }
  ];

  // If there's no search query, return all items
  if (!searchQuery) {
    return items.slice(0, 10);
  }

  return items
    .filter((item) => 
      item.title.toLowerCase().startsWith(searchQuery.toLowerCase())
    )
    .slice(0, 10);
};

export default getSuggestionItems;