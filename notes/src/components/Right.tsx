import { Note } from "../App";
import RelevantArticles from "./RelevantArticles";

function Right({ notes, active }: { notes: Note[]; active: number }) {
  return (
    <div className="grid grid-rows-[auto_max-content] border-l-1 border-white/15 px-4 pt-10 max-h-screen overflow-hidden">
      <div className="flex flex-col w-[16rem]">
        <h1 className="text-orange-600 text-xl font-bold">
          {notes[active].title}
        </h1>
        <h2 className="italic text-white/40 text-sm">
          <a href="">[Author 1]</a>
          <a href="">[Author 2]</a>
          <a href="">[Author 3]</a>
        </h2>
        <p className="text-sm text-ellipsis">
          {notes[active].data.text.slice(0, 700)}...
        </p>
      </div>
      <RelevantArticles articles={notes[active].data.related} />
    </div>
  );
}

export default Right;
