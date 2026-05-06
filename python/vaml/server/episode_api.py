from fastapi import FastAPI
from vaml.buffer import UpdateBatch

app = FastAPI()

@app.get("/episode/{episode_id}")
def read_item(episode_id: int):
    ub = UpdateBatch.load_npz("./offline_data/episodes_0.npz")

    return {"tokens": ub.context.tolist()}
