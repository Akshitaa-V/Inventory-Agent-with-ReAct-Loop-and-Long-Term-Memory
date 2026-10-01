"""
InnKube LLM API client.
"""

import requests

from agent.config import Config


class LLMClient:
    def __init__(self, config: Config):
        self.config = config

    def get_next_step(self, messages: list, tools: list | None = None) -> dict:
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools

        response = requests.post(
            f"{self.config.base_url}{self.config.endpoint}",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json=payload,
            timeout=60,
        )
        if not response.ok:
            print("---- DEBUG: server error response ----")
            print(response.status_code, response.text)
            print("---------------------------------------")
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]["message"]

        # OpenAI-compatible APIs (InnKube included) return token counts in
        # a top-level "usage" object. Not every deployment is guaranteed to
        # include it, so this is treated as optional -- callers check for
        # None rather than assuming it's always present ("token consumption
        # where available", per the Week 3 handout's own wording).
        usage = data.get("usage") or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")

        tool_calls = choice.get("tool_calls")
        if tool_calls:
            call = tool_calls[0]
            return {
                "type": "tool_call",
                "id": call["id"],
                "tool_name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
                "model": self.config.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        return {
            "type": "final_answer",
            "content": choice.get("content", ""),
            "model": self.config.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }