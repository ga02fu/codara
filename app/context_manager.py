from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_TOTAL_BUDGET = 12000
DEFAULT_SECTION_BUDGETS = {
    "system_prefix": 3000,
    "workspace": 2500,
    "detail_memory": 2400,
    "general_memory": 4100,
}
DEFAULT_REDUCTION_ORDER = (
    "general_memory",
    "detail_memory",
    "workspace",
    "system_prefix",
)


def _tail_clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


@dataclass
class SectionRender:
    raw: str
    budget: int
    rendered: str


@dataclass
class ContextManager:
    """
    提示词组装器与上下文瘦身控制器。
    对齐 pico 文档 06：分层瘦身而不是整段粗暴截断。
    """

    max_total_chars: int = DEFAULT_TOTAL_BUDGET
    section_budgets: dict[str, int] | None = None
    reduction_order: tuple[str, ...] = DEFAULT_REDUCTION_ORDER

    def __post_init__(self) -> None:
        base = dict(DEFAULT_SECTION_BUDGETS)
        if self.section_budgets:
            for key, value in self.section_budgets.items():
                base[str(key)] = int(value)
        self.section_budgets = base

    def _floors(self) -> dict[str, int]:
        return {
            k: max(20, int(v) // 4) for k, v in (self.section_budgets or {}).items()
        }

    def build_prompt(
        self,
        system_prefix: str,
        workspace_text: str,
        detail_memory_text: str,
        general_memory: list[dict[str, str]],
        user_message: str,
    ) -> tuple[str, dict[str, Any]]:
        budgets = self.section_budgets or dict(DEFAULT_SECTION_BUDGETS)
        general_memory_raw = self._raw_general_memory(general_memory)
        render_map: dict[str, SectionRender] = {
            "system_prefix": SectionRender(
                raw=system_prefix,
                budget=int(budgets["system_prefix"]),
                rendered=_tail_clip(system_prefix, int(budgets["system_prefix"])),
            ),
            "workspace": SectionRender(
                raw=workspace_text,
                budget=int(budgets["workspace"]),
                rendered=_tail_clip(workspace_text, int(budgets["workspace"])),
            ),
            "detail_memory": SectionRender(
                raw=detail_memory_text,
                budget=int(budgets["detail_memory"]),
                rendered=_tail_clip(detail_memory_text, int(budgets["detail_memory"])),
            ),
            "general_memory": SectionRender(
                raw=general_memory_raw,
                budget=int(budgets["general_memory"]),
                rendered=self._render_general_memory(
                    general_memory, int(budgets["general_memory"])
                ),
            ),
        }

        reductions: list[dict[str, int | str]] = []
        prompt = self._assemble(render_map, user_message)
        floors = self._floors()
        while len(prompt) > self.max_total_chars:
            overflow = len(prompt) - self.max_total_chars
            reduced = False
            for section in self.reduction_order:
                current_budget = int(render_map[section].budget)
                floor = int(floors.get(section, 20))
                if current_budget <= floor:
                    continue
                next_budget = max(floor, current_budget - overflow)
                if next_budget >= current_budget:
                    continue
                reductions.append(
                    {
                        "section": section,
                        "before": current_budget,
                        "after": next_budget,
                        "overflow": overflow,
                    }
                )
                render_map[section].budget = next_budget
                if section == "general_memory":
                    render_map[section].rendered = self._render_general_memory(
                        general_memory, next_budget
                    )
                else:
                    render_map[section].rendered = _tail_clip(
                        render_map[section].raw, next_budget
                    )
                prompt = self._assemble(render_map, user_message)
                reduced = True
                break
            if not reduced:
                break

        meta = {
            "sections": {k: len(v.rendered) for k, v in render_map.items()}
            | {"request": len(user_message.strip())},
            "section_budgets": {k: v.budget for k, v in render_map.items()},
            "budget_reductions": reductions,
            "prompt_chars": len(prompt),
            "cache_reuse_enabled": False,
        }
        return prompt, meta

    def _assemble(self, render_map: dict[str, SectionRender], user_message: str) -> str:
        return (
            "你是 Codara，一个代码助手。\n"
            "你必须遵循单工具、单步推进、禁止重复探测的执行策略。\n"
            "用户输入通常已被重写为“用户请求 + 子任务”结构，请优先按子任务编号执行。\n"
            '输出只能是 <tool>{"name":"...","args":{...}}</tool>、<retry>...</retry> 或 <final>...</final>。\n'
            "当本轮无需工具但仍需继续推理时，使用 <retry>...</retry>。\n"
            "若请求包含多个子任务，允许部分完成；必须在 final 中按编号说明完成项和失败项。\n"
            "若用户请求是创建/写入，确认目标路径后应尽快使用 write_file/patch_file 完成，不可在 list_files 循环。\n\n"
            "=== SYSTEM ===\n"
            f"{render_map['system_prefix'].rendered}\n\n"
            "=== WORKSPACE ===\n"
            f"{render_map['workspace'].rendered}\n\n"
            "=== DETAIL MEMORY ===\n"
            f"{render_map['detail_memory'].rendered}\n\n"
            "=== GENERAL MEMORY ===\n"
            f"{render_map['general_memory'].rendered}\n\n"
            "=== USER REQUEST ===\n"
            f"{user_message.strip()}\n"
        )

    def _raw_general_memory(self, general_memory: list[dict[str, str]]) -> str:
        if not general_memory:
            return "(empty)"
        lines: list[str] = []
        for item in general_memory:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    def _render_general_memory(
        self, general_memory: list[dict[str, str]], budget: int
    ) -> str:
        if not general_memory:
            return "(empty)"

        recent_window = 6
        recent_start = max(0, len(general_memory) - recent_window)
        rendered_entries: list[str] = []

        for idx in reversed(range(len(general_memory))):
            item = general_memory[idx]
            role = item.get("role", "unknown")
            content = str(item.get("content", ""))
            line_limit = 900 if idx >= recent_start else 80
            line = f"[{role}] {_tail_clip(content, line_limit)}"
            candidate = [line, *rendered_entries]
            candidate_text = "\n".join(candidate)
            if len(candidate_text) <= budget:
                rendered_entries = candidate
                continue

            if idx >= recent_start:
                short_line = f"[{role}] {_tail_clip(content, max(20, line_limit // 3))}"
                candidate = [short_line, *rendered_entries]
                candidate_text = "\n".join(candidate)
                if len(candidate_text) <= budget:
                    rendered_entries = candidate

        if not rendered_entries:
            return _tail_clip(self._raw_general_memory(general_memory), budget)
        return "\n".join(rendered_entries)
