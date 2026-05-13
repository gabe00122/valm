import * as v from "valibot";
import { query } from "$app/server";

export const getEpisode = query(v.number(), async (episodeId) => {
    // const episodeId = 0;
    const response = await fetch(`http://127.0.0.1:8000/episode/${episodeId}`);
    const json: { tokens: string[] } = await response.json();

    return json;
});
