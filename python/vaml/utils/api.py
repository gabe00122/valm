from typing import Literal

import httpx
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


def completion(
    messages: list[Message],
    base_url: str,
    api_key: str,
    model: str,
    reasoning_enabled: bool = True,
):
    message_dicts = [message.model_dump() for message in messages]

    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": model,
            "messages": message_dicts,
            "reasoning": {"enabled": reasoning_enabled},
        },
        timeout=None,
    )
    response_json = response.json()
    response_message = response_json["choices"][0]["message"]

    return Message.model_validate(response_message)
