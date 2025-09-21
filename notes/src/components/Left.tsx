import { UserRound } from "lucide-react";
import { Note } from "../App";

function Left({
  notes,
  bookmarked,
  setActive,
}: {
  notes: Note[];
  bookmarked: { title: string; time: Date }[];
  setActive: (index: number) => void;
}) {
  return (
    <div className="flex flex-col justify-between border-r-1 border-white/15 p-4">
      <div className="flex flex-col gap-8">
        <div className="grid grid-cols-[min-content_auto] gap-2">
          <div>
            <div className="shadow-[0_0_0_1px_#fff_inset] rounded-full overflow-hidden">
              <UserRound className="h-6 w-6" />
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <h1 className="text-sm">Yuvraj Chaudhary</h1>
            <p className="text-white/40 text-xs">Student Account</p>
          </div>
        </div>
        <div className="text-sm flex flex-col gap-1">
          <div className="px-1 py-2">
            <h2 className="font-bold">Recently Viewed</h2>
          </div>
          <ul>
            {notes.map((item, i) => (
              <li
                key={item.title}
                className="p-2 hover:bg-white/5 rounded-md cursor-pointer italic"
                onClick={() => {
                  setActive(-1);
                  setTimeout(() => {
                    setActive(i);
                  }, 1500);
                }}
              >
                <h3
                  className="text-ellipsis overflow-hidden whitespace-nowrap"
                  title={item.title}
                >
                  {item.title}
                </h3>
                <p className="text-white/40 text-xs">
                  {item.time.toLocaleString()}
                </p>
              </li>
            ))}
            {bookmarked.map((item, i) => (
              <li
                key={item.title}
                className="p-2 hover:bg-white/5 rounded-md cursor-pointer italic"
                onClick={() => setActive(i)}
              >
                <h3
                  className="text-ellipsis overflow-hidden whitespace-nowrap"
                  title={item.title}
                >
                  {item.title}
                </h3>
                <p className="text-white/40 text-xs">
                  {item.time.toLocaleString()}
                </p>
              </li>
            ))}
          </ul>
        </div>
      </div>
      <div>
        <p className="!font-[Mortend-Bold] text-center text-white/40">
          sen_dex
        </p>
      </div>
    </div>
  );
}

export default Left;
