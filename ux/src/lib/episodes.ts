export interface EncodedEpisode {
    tokens: string[];
    tokenIds: number[];
    logProbs: string;
    values: string;
    rewards: string;
    policyMask: string;
    updateMetrics: Record<string, string>;
}

export interface Episode {
    tokens: string[];
    tokenIds: number[];
    logProbs: Float32Array;
    values: Float32Array;
    rewards: Float32Array;
    policyMask: Uint8Array;
    updateMetrics: Record<string, Float32Array>;
}
