import * as v from "valibot";
import { query } from "$app/server";
import type { EncodedEpisode } from "$lib/episodes";

export const getEpisode = query(v.number(), async (episodeId) => {
  const response = await fetch(`http://127.0.0.1:8000/episode/${episodeId}`);
  const json: EncodedEpisode = await response.json();
  return json;
});
