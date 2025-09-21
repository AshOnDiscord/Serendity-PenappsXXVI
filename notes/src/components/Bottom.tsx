import { Search } from "lucide-react";
import EditorWrapper from "./EditorWrapper";
import { useState } from "react";

function Bottom() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <footer className="flex justify-between p-5 border-t-1 border-white/15">
        <div>
          <div className="flex bg-white/15 items-center rounded-full px-3 py-2 gap-2 has-[:focus]:ring-1 transition has-[:focus]:ring-white/40 ring-white/0 ring-1">
            <Search className="h-4 w-4" />
            <input
              type="text"
              placeholder="Search a concept, idea, or link."
              className="p-0 bg-transparent placeholder:text-white/40 border-0 outline-none ring-0 "
            />
          </div>
        </div>
        <div className="flex">
          <button
            className="bg-white/10 rounded-xl text-white/40 font-bold text-xs py-3.5 px-8"
            onClick={() => setIsOpen(true)}
          >
            Take a Note
          </button>
        </div>
      </footer>
      <EditorWrapper isOpen={isOpen} setIsOpen={setIsOpen} />
    </>
  );
}

export default Bottom;
