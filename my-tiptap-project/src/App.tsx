import "./styles.scss";
import Tiptap from "./tiptap";

const App: React.FC = () => {
  return (
    <div className="App">
      <h1>Hello Vite + Electron</h1>
      <h2>Start editing to see some magic happen!</h2>
      <Tiptap />
    </div>
  );
};

export default App;