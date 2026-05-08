from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class ModelError(Exception):
    pass


@dataclass
class CompletionResult:
    text: str
    usage: dict[str, Any]
    raw: dict[str, Any]


class DeepSeekChatClient:
    """
    DeepSeek 单后端模型客户端。
    仅支持 deepseek-chat，保持接口统一，方便后续扩展而不改 runtime 主逻辑。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout: int = 60,
        temperature: float = 0.2,
    ):
        if not api_key.strip():
            raise ModelError("DEEPSEEK_API_KEY 未设置。")
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.supports_prompt_cache = False
        self.last_completion_metadata: dict[str, Any] = {}

    def complete(self, prompt: str, max_new_tokens: int = 1200) -> CompletionResult:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_text = resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            raise ModelError(f"DeepSeek HTTP 错误: {e.code} - {detail}") from e
        except urllib.error.URLError as e:
            raise ModelError(f"DeepSeek 网络错误: {e}") from e
        except Exception as e:
            raise ModelError(f"DeepSeek 未知错误: {e}") from e

        try:
            raw = json.loads(raw_text)
            text = raw["choices"][0]["message"]["content"]
            usage = raw.get("usage", {})
        except Exception as e:
            raise ModelError(f"DeepSeek 响应解析失败: {e}; raw={raw_text[:500]}") from e

        self.last_completion_metadata = {
            "model": self.model,
            "usage": usage,
        }
        return CompletionResult(text=text, usage=usage, raw=raw)