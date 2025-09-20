import { useEditor, EditorContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Commands from "./suggestion/commands";
import getSuggestionItems from "./suggestion/items";
import renderItems from "./suggestion/renderItems";
import './index.css'; // Import the CSS file here

const logMessage = (val: any): void => {
  console.log(`show / hide the image upload modal`);
};

const Tiptap: React.FC = () => {
  const editor = useEditor({
    extensions: [
      StarterKit,
      Commands.configure({
        suggestion: {
          items: getSuggestionItems,
          render: renderItems
        }
      })
    ],
    content: "<p>Use / command to see different options</p> "
  });

  return <EditorContent editor={editor} className="editor" />;
};

export default Tiptap;