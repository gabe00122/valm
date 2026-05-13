from fastapi import FastAPI
from vaml.buffer import UpdateBatch
from vaml.util import load_tokenizer
from functools import cache


app = FastAPI()

tokenizer = load_tokenizer("./base-models/Qwen/Qwen3-4B-Instruct-2507")
episodes_per_file = 100

@cache
def get_update_batch(chunk_id: int) -> UpdateBatch:
    return UpdateBatch.load_npz(f"./offline_data/episodes_{chunk_id}.npz")


@app.get("/episode/{episode_id}")
def read_item(episode_id: int):
    chunk = episode_id // episodes_per_file
    ub = get_update_batch(chunk)

    length = ub.context_length[episode_id]
    context = ub.context[episode_id, :length].tolist()
    toks = [tokenizer.decode([token_id]) for token_id in context]

    return {"tokens": toks}
