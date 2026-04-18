from typing import Literal
from pydantic import BaseModel
import httpx

base_url = "http://localhost:8080"

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

def completion(messages: list[Message]):
    message_dicts = [message.model_dump() for message in messages]

    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": "Bearer no-key"},
        json={"model": "blank", "messages": message_dicts},
        timeout=None
    )
    response_json = response.json()
    response_message = response_json["choices"][0]["message"]

    return Message.model_validate(response_message)
