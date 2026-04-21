import json

from transformers import PreTrainedTokenizerFast
from vaml.buffer import UpdateBatch


def episode_to_jsonl(
    jsonl_path: str, episode: UpdateBatch, tokenizer: PreTrainedTokenizerFast
):
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(episode.context.shape[0]):
            length = episode.kv_cache_lengths[i] + 1
            ids = episode.context[i, :length].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=False)
            toks = tokenizer.convert_ids_to_tokens(ids)

            ep = {
                "text": text,
                "tokens": toks,
                "token_ids": ids,
                "rewards": episode.rewards[i, :length].tolist(),
                "values": episode.values[i, :length].tolist(),
                "log_probs": episode.log_probs[i, :length].tolist(),
                "policy_mask": episode.policy_mask[i, :length].tolist(),
            }
            f.write(json.dumps(ep) + "\n")
