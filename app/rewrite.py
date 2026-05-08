from __future__ import annotations

import re

from .models import DeepSeekChatClient


class RequestRewriter:
    """
    请求重写器。
    在进入主循环前，将自然语言请求重写为“用户请求 + 子任务”结构，
    以提升模型执行稳定性。
    """

    def __init__(
        self,
        model_client: DeepSeekChatClient,
        enabled: bool = True,
        max_new_tokens: int = 420,
    ):
        self.model_client = model_client
        self.enabled = enabled
        self.max_new_tokens = max_new_tokens

    def rewrite(self, user_message: str) -> str:
        text = str(user_message).strip()
        if not self.enabled or not text:
            return text

        # 已经是目标格式时直接复用，避免重复重写导致漂移。
        if self._looks_rewritten(text):
            return text

        prompt = self._build_prompt(text)
        completion = self.model_client.complete(
            prompt=prompt, max_new_tokens=self.max_new_tokens
        )
        rewritten = completion.text.strip()

        if not rewritten:
            return text
        if not self._looks_rewritten(rewritten):
            return self._fallback_format(text)
        return rewritten

    def _looks_rewritten(self, text: str) -> bool:
        has_head = "用户请求" in text and "子任务" in text
        has_numbered = bool(re.search(r"(?m)^\s*\d+[\.|、]\s*\S+", text))
        return has_head and has_numbered

    def _fallback_format(self, user_message: str) -> str:
        return f"用户请求：\n{user_message.strip()}\n\n子任务：\n1. 按用户请求完成主要目标。"

    def _build_prompt(self, user_message: str) -> str:
        return (
            "你是请求重写器。请将用户原始请求重写成结构化输入，供编码 Agent 执行。\n"
            "只输出最终重写结果，不要解释。\n"
            "输出格式必须严格如下：\n"
            "用户请求：\n"
            "<对原始请求的一句话归纳，不得改变意图>\n\n"
            "子任务：\n"
            "1. <子任务1>\n"
            "2. <子任务2>\n"
            "...\n\n"
            "约束：\n"
            "1) 子任务按可执行顺序排列，粒度适中。\n"
            "2) 不得凭空新增与原请求无关的目标。\n"
            "3) 原请求只有单任务时，也要输出 1 条子任务。\n"
            "4) 禁止输出 XML/HTML 标签，尤其是 <tool>/<final>。\n"
            "5) 使用简体中文。\n\n"
            f"原始请求：\n{user_message.strip()}\n"
        )
