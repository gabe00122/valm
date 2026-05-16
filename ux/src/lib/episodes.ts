export interface EncodedEpisode {
    tokens: string[];
    logProbs: string;
    updateMetrics: Record<string, string>;
}

export interface Episode {
    tokens: string[];
    logProbs: Float32Array;
    updateMetrics: Record<string, Float32Array>;
}
