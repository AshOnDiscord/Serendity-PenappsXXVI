import { useEditor, EditorContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import { SimpleEditor } from '../@/components/tiptap-templates/simple/simple-editor'
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
  const [editorInstance, setEditorInstance] = useState<any>(null);
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
    if (!editorInstance) return;
    const json = editorInstance.getJSON();
    setJsonContent(json);               // Show JSON on screen
    window.electronAPI.saveJSON(json);  // Save JSON to disk
  }

  const importJSON = async () => {
    const data = await window.electronAPI.openJSON();
    if (data && editorInstance) {
      editorInstance.commands.setContent(data);
      setJsonContent(data);
    }
  }

  return (
  <div>
      <div style={{ marginBottom: '10px' }}>
        <button onClick={exportJSON} style={{ marginRight: '10px' }}>
          Save as JSON
        </button>
        <button onClick={importJSON}>Import JSON</button>
      </div>
    
      <SimpleEditor onEditorReady={(editor: any) => setEditorInstance(editor)} />

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