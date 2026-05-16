from fastapi import FastAPI
from vaml.buffer import UpdateBatch
from vaml.util import load_tokenizer
from functools import cache
import numpy as np
import base64


app = FastAPI()

tokenizer = load_tokenizer("./base-models/Qwen/Qwen3-4B-Instruct-2507")
episodes_per_file = 100

@cache
def get_update_batch(chunk_id: int) -> UpdateBatch:
    return UpdateBatch.load_npz(f"./value_trace/episodes_{chunk_id}.npz")

def base64_encode(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode('utf-8')

@app.get("/episode/{episode_id}")
def read_item(episode_id: int):
    chunk = episode_id // episodes_per_file
    ub = get_update_batch(chunk)

    length = ub.context_length[episode_id]
    context = ub.context[episode_id, :length].tolist()
    toks = [tokenizer.decode([token_id]) for token_id in context]

    print(ub.log_probs[episode_id, :length])
    log_probs = base64_encode(ub.log_probs[episode_id, :length])

    update_metrics = {name: base64_encode(value[episode_id, :length]) for name, value in ub.update_metrics.items()}
    print(update_metrics)

    return {"tokens": toks, "logProbs": log_probs, "updateMetrics": update_metrics}
