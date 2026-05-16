import type { EncodedEpisode, Episode } from "./episodes";

function base64DecodeBytes(data: string): Uint8Array {
    const binaryString = atob(data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    return bytes;
}

export function base64Decode(data: string): Float32Array {
    const bytes = base64DecodeBytes(data);
    return new Float32Array(bytes.buffer);
}

export function base64DecodeBool(data: string): Uint8Array {
    return base64DecodeBytes(data);
}

export function decodeEpisodes(episode: EncodedEpisode): Episode {
    const { tokens, tokenIds, logProbs, values, rewards, policyMask, updateMetrics } =
        episode;

    const um = Object.fromEntries(
        Object.entries(updateMetrics).map(([name, value]) => [
            name,
            base64Decode(value),
        ]),
    );

    return {
        tokens,
        tokenIds,
        logProbs: base64Decode(logProbs),
        values: base64Decode(values),
        rewards: base64Decode(rewards),
        policyMask: base64DecodeBool(policyMask),
        updateMetrics: um,
    };
}
