// src/Tiptap.tsx
import { useEditor, EditorContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Commands from "./plugins/slashCommands/commands";
import getSuggestionItems from "./plugins/slashCommands/items";
import renderItems from "./plugins/slashCommands/renderItems";
import {
  Bold,
  Heading1,
  Heading2,
  Heading3,
  Highlighter,
  Italic,
  Link,
  List,
  ListOrdered,
  Redo2,
  Strikethrough,
  TextQuote,
  Underline,
  Undo2,
  Superscript as SuperscriptIcon,
  Subscript as SubscriptIcon,
} from "lucide-react";
import { useState } from "react";
import { Node } from "@tiptap/pm/model";
import { Selection } from "@tiptap/pm/state";
import Superscript from "@tiptap/extension-superscript";
import Subscript from "@tiptap/extension-subscript";

function App() {
  const editor = useEditor({
    editorProps: {
      attributes: {
        class:
          "editor prose prose-tiptap mx-auto focus:outline-none p-[3rem_3rem_30vh] prose-h1:text-2xl prose-h1:mt-12 prose-h1:mb-0 prose-h1:font-bold prose-h2:text-xl prose-h2:mt-10 prose-h2:mb-0 prose-h2:font-bold prose-h3:text-lg prose-h3:mt-8 prose-h3:mb-0 prose-h3:font-semibold prose-p:mt-5 prose-p:mb-0" +
          " prose-li:ps-0 prose-hr:my-9",
      },
    },
    extensions: [
      StarterKit,
      Commands.configure({
        suggestion: {
          items: getSuggestionItems,
          render: renderItems,
        },
      }),
      Superscript,
      Subscript,
    ], // define your extension array
    content: "<p>Hello World!</p>", // initial content
  });

  const [editorState, setEditorState] = useState<{
    isEditable: boolean;
    currentSelection: Selection;
    heading: Node | null;
    isBlockquote: boolean;
    isBold: boolean;
    isItalic: boolean;
    isUnderline: boolean;
    isStrike: boolean;
    isBulletList: boolean;
    isOrderedList: boolean;
    isSuperscript: boolean;
    isSubscript: boolean;
  } | null>(null);

  editor.on("update", () => {
    const { $from } = editor.state.selection;

    setEditorState({
      isEditable: editor.isEditable,
      currentSelection: editor.state.selection,
      // currentContent: editor.getJSON(),
      heading: $from.node($from.depth),

      // you can add more state properties here e.g.:
      isBlockquote: editor.isActive("blockquote"),
      isBold: editor.isActive("bold"),
      isItalic: editor.isActive("italic"),
      isUnderline: editor.isActive("underline"),
      isStrike: editor.isActive("strike"),
      isBulletList: editor.isActive("bulletList"),
      isOrderedList: editor.isActive("orderedList"),
      isSuperscript: editor.isActive("superscript"),
      isSubscript: editor.isActive("subscript"),
    });
  });

  const toggleHeading = (level: 1 | 2 | 3) => {
    if (!editor) return;
    if (editorState?.heading?.attrs?.level === level) {
      editor.chain().focus().setParagraph().run();
      return;
    }
    editor.chain().focus().setHeading({ level }).run();
  };

  return (
    <>
      <nav className="flex justify-center border-b border-[rgba(232,232,253,0.05)] editor-navbar gap-1 items-center">
        <div>
          <button
            disabled={!editor.can().undo()}
            onClick={() => editor?.chain().focus().undo().run()}
          >
            <Undo2 />
          </button>
          <button
            disabled={!editor.can().redo()}
            onClick={() => editor?.chain().focus().redo().run()}
          >
            <Redo2 />
          </button>
        </div>
        <span />
        <div className="heading-btns">
          <button
            onClick={() => toggleHeading(1)}
            className={editorState?.heading?.attrs?.level === 1 ? "active" : ""}
          >
            <Heading1 />
          </button>
          <button
            onClick={() => toggleHeading(2)}
            className={editorState?.heading?.attrs?.level === 2 ? "active" : ""}
          >
            <Heading2 />
          </button>
          <button
            onClick={() => toggleHeading(3)}
            className={editorState?.heading?.attrs?.level === 3 ? "active" : ""}
          >
            <Heading3 />
          </button>
        </div>
        <span />
        <div>
          <button
            onClick={() => editor?.chain().focus().toggleBulletList().run()}
            className={editorState?.isBulletList ? "active" : ""}
          >
            <List />
          </button>
          <button
            onClick={() => editor?.chain().focus().toggleOrderedList().run()}
            className={editorState?.isOrderedList ? "active" : ""}
          >
            <ListOrdered />
          </button>
        </div>
        <span />
        <div>
          <button
            onClick={() => editor?.chain().focus().toggleBlockquote().run()}
            className={editorState?.isBlockquote ? "active" : ""}
          >
            <TextQuote />
          </button>
          <button
            onClick={() => editor?.chain().focus().toggleBold().run()}
            className={editorState?.isBold ? "active" : ""}
          >
            <Bold />
          </button>
          <button
            onClick={() => editor?.chain().focus().toggleItalic().run()}
            className={editorState?.isItalic ? "active" : ""}
          >
            <Italic />
          </button>
          <button
            onClick={() => editor?.chain().focus().toggleStrike().run()}
            className={editorState?.isStrike ? "active" : ""}
          >
            <Strikethrough />
          </button>
          <button
            onClick={() => editor?.chain().focus().toggleUnderline().run()}
            className={editorState?.isUnderline ? "active" : ""}
          >
            <Underline />
          </button>
          <button>
            <Highlighter />
          </button>
          <button>
            <Link />
          </button>
        </div>
        <span />
        <div>
          <button
            onClick={() => editor?.chain().focus().toggleSuperscript().run()}
            className={editorState?.isSuperscript ? "active" : ""}
          >
            <SuperscriptIcon />
          </button>
          <button
            onClick={() => editor?.chain().focus().toggleSubscript().run()}
            className={editorState?.isSubscript ? "active" : ""}
          >
            <SubscriptIcon />
          </button>
        </div>
      </nav>
      <EditorContent editor={editor} />
      {/* <FloatingMenu editor={editor}>This is the floating menu</FloatingMenu>
      <BubbleMenu editor={editor}>This is the bubble menu</BubbleMenu> */}
    </>
  );
}

export default App;
