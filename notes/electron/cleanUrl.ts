import { filters } from "./filter";
import { minimatch } from "minimatch";

export default function cleanURL(url: string) {
  const parsed = new URL(url);
  const clean = parsed.origin + parsed.pathname;
  const searchParams = parsed.searchParams;

  filterLoop: for (const filter of filters) {
    if (searchParams.size === 0) break;
    // check blacklist
    const blacklists = filter.exclude || [];
    for (const blacklist of blacklists) {
      if (minimatch(clean, blacklist)) {
        // url is blacklisted, skip this filter
        continue filterLoop;
      }
    }
    let matched = false;
    for (const pattern of filter.include) {
      // if (pattern.includes(".com")) debugger;
      if (minimatch(clean, pattern)) {
        matched = true;
        break;
      }
    }
    if (!matched) {
      // url is not whitelisted, skip this filter
      continue filterLoop;
    }
    // remove params
    for (const param of filter.params) {
      searchParams.delete(param);
    }
  }

  if (searchParams.size === 0) {
    return clean;
  }
  return clean + "?" + searchParams.toString();
}
