import type { EncodedEpisode, Episode } from "./episodes";

export function base64Decode(data: string): Float32Array {
    const binaryString = atob(data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    return new Float32Array(bytes.buffer);
}

export function decodeEpisodes(episode: EncodedEpisode): Episode {
    const { tokens, logProbs, updateMetrics } = episode;

    const um = Object.fromEntries(
        Object.entries(updateMetrics).map(([name, value]) => [
            name,
            base64Decode(value),
        ]),
    );

    return { tokens, logProbs: base64Decode(logProbs), updateMetrics: um };
}
