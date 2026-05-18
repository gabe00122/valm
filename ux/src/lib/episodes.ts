export interface EncodedEpisode {
  tokens: string[];
  tokenMetrics: Record<string, string>;
}

export interface Episode {
  tokens: string[];
  tokenMetrics: Record<string, Float32Array>;
}
