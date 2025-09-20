import { useEditor, EditorContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Document from '@tiptap/extension-document'
import Paragraph from '@tiptap/extension-paragraph'
import Text from '@tiptap/extension-text'
import Commands from "./suggestion/commands";
import getSuggestionItems from "./suggestion/items";
import renderItems from "./suggestion/renderItems";
import './index.css'; // Import the CSS file here
import React, { useState } from "react";

const Tiptap: React.FC = () => {
  const [jsonContent, setJsonContent] = useState<any>(null)
  const editor = useEditor({
    extensions: [
      StarterKit,
      // Document, Paragraph, Text,
      Commands.configure({
        suggestion: {
          items: getSuggestionItems,
          render: renderItems
        }
      })
    ],
    content: "<p>Use / command to see different options</p> "
  });

  const exportJSON = () => {
    if (!editor) return
    const json = editor.getJSON()
    setJsonContent(json)              // Show JSON on screen
    window.electronAPI.saveJSON(json) // Save JSON to disk
  }

  return (
  <div>
    <button onClick={exportJSON} style={{ marginBottom: '20px' }}>
      Save as JSON
    </button>
    
    <EditorContent editor={editor} className="editor" />

      {jsonContent && (
        <div style={{ marginTop: '20px' }}>
          <h3>Editor JSON:</h3>
          <pre
            style={{
              background: 'black',
              padding: '10px',
              borderRadius: '5px',
              overflowX: 'auto',
            }}
          >
            {JSON.stringify(jsonContent, null, 2)}
          </pre>
        </div>
      )}
  </div>)
};

export default Tiptap;