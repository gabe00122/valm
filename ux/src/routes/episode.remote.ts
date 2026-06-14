import * as v from "valibot";
import { query } from "$app/server";
import type { EncodedEpisode } from "$lib/episodes";

const host = "http://127.0.0.1:8000";

export const getRuns = query(async () => {
  const response = await fetch(`${host}/runs`);
  const json: { runs: string[] } = await response.json();
  return json.runs;
});

export const getRunInfo = query(v.string(), async (run) => {
  const response = await fetch(`${host}/runs/${run}`);
  const json: { name: string; episodeCount: number } = await response.json();
  return json;
});

export const getEpisode = query(
  v.object({ run: v.string(), episodeId: v.number() }),
  async ({ run, episodeId }) => {
    const response = await fetch(`${host}/runs/${run}/episode/${episodeId}`);
    const json: EncodedEpisode = await response.json();
    return json;
  },
);
