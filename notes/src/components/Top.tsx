import { PanelLeftDashed, Star } from "lucide-react";

function Top() {
  return (
    <nav className="px-7 py-5 flex justify-between border-b-1 border-white/15 items-center">
      <div className="flex gap-5">
        <div className="flex gap-2">
          <button className="p-1.5 pb-0 cursor-pointer">
            <PanelLeftDashed className="h-4 w-4 pointer-none" />
          </button>
          <button className="p-1.5 pb-0 cursor-pointer">
            <Star className="h-4 w-4 pointer-none" />
          </button>
        </div>
        <div className="flex gap-5">
          <h2 className="text-white/40">Science Research</h2>
          <span className="text-white/10">/</span>
          <h1>Arxiv Database</h1>
        </div>
      </div>
      <div></div>
    </nav>
  );
}
export default Top;
