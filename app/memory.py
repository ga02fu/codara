from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable


TOP_K_RELEVANT_NOTES = 5


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class LayeredMemory:
    """
    结构化会话记忆。
    轻量化，结构化，与工具链绑定
    工具执行结果先做 LLM 摘要，再写入 relevant_notes
    relevant_notes 在写入时完成去重与 top_k 裁剪
    记忆分层：
    - task_summary: 任务摘要
    - recent_files: 最近打开的文件列表
    - file_summaries: 文件摘要字典，key 为文件路径，value 为摘要字符串
    - relevant_notes: 工具执行记录摘要列表（去重后）
    """

    task_summary: str = ""
    recent_files: list[str] = field(default_factory=list)
    file_summaries: dict[str, dict[str, str]] = field(default_factory=dict)
    relevant_notes: list[str] = field(default_factory=list)
    summarize_tool_fn: Callable[[str, dict[str, Any], str], str] | None = field(
        default=None, repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_summary": self.task_summary,
            "recent_files": self.recent_files,
            "file_summaries": self.file_summaries,
            "relevant_notes": self.relevant_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayeredMemory":
        detail_memory = data if isinstance(data, dict) else {}

        file_summaries = detail_memory.get("file_summaries", {})
        if not isinstance(file_summaries, dict):
            file_summaries = {}

        raw_notes = detail_memory.get("relevant_notes", [])
        normalized_notes: list[str] = []
        for item in raw_notes or []:
            if isinstance(item, dict):
                text = str(item.get("message", "")).strip()
            else:
                text = str(item).strip()
            if text:
                normalized_notes.append(text)

        normalized_notes = cls._dedupe_trim_notes(
            normalized_notes, TOP_K_RELEVANT_NOTES
        )

        recent_files = detail_memory.get("recent_files", [])
        if not isinstance(recent_files, list):
            recent_files = []

        return cls(
            task_summary=detail_memory.get("task_summary", ""),
            recent_files=recent_files,
            file_summaries=file_summaries,
            relevant_notes=normalized_notes,
        )

    @staticmethod
    def _note_identity(text: str) -> str:
        s = str(text).strip()
        if not s:
            return ""

        m_file = re.match(r"^文件\s*([^\s：:]+)\s*主要内容", s)
        if m_file:
            return f"file:{m_file.group(1)}"

        if s.startswith("list_files"):
            if "当前目录" in s:
                return "list_files:."
            m_dir = re.search(r"([A-Za-z0-9_./-]+)\s*目录", s)
            if m_dir:
                return f"list_files:{m_dir.group(1)}"
            return "list_files"

        return s

    @classmethod
    def _dedupe_trim_notes(cls, notes: list[str], limit: int) -> list[str]:
        seen: set[str] = set()
        picked_rev: list[str] = []
        for item in reversed(notes):
            text = str(item).strip()
            if not text:
                continue
            key = cls._note_identity(text)
            if key in seen:
                continue
            seen.add(key)
            picked_rev.append(text)
            if len(picked_rev) >= limit:
                break
        return list(reversed(picked_rev))

    def set_task_summary(self, text: str) -> None:
        self.task_summary = text.strip()[:300]

    def add_recent_file(self, path: str) -> None:
        path = path.strip()
        if not path:
            return
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        self.recent_files = self.recent_files[:8]

    def upsert_file_summary(self, path: str, content_preview: str) -> None:
        key = path.strip()
        if not key:
            return
        preview = content_preview[:500]
        self.file_summaries[key] = {
            "summary": preview,
            "content_hash": _sha1(content_preview),
        }
        if len(self.file_summaries) > 20:
            # 简单淘汰策略：按插入顺序淘汰最早项
            first_key = next(iter(self.file_summaries))
            if first_key != key:
                self.file_summaries.pop(first_key, None)

    def add_relevant_note(self, message: str) -> None:
        text = message[:300]
        if not text.strip():
            return
        self.relevant_notes = self._dedupe_trim_notes(
            [*self.relevant_notes, text], TOP_K_RELEVANT_NOTES
        )

    def add_note(self, kind: str, message: str) -> None:
        # 兼容旧调用签名。
        _ = kind
        self.add_relevant_note(message)

    def update_after_tool(
        self, tool_name: str, args: dict[str, Any], result: str
    ) -> None:
        """
        在工具执行后更新记忆
        """
        note_message = f"{tool_name}: {result[:200]}"
        if self.summarize_tool_fn is not None:
            try:
                summarized = str(
                    self.summarize_tool_fn(tool_name, args, result)
                ).strip()
                if summarized:
                    note_message = summarized[:300]
            except Exception:
                # 摘要失败时不要回填原始结果，避免代码片段污染记忆。
                path = str(args.get("path", "")).strip()
                if path:
                    note_message = f"{tool_name} {path} 已执行，结果已记录。"
                else:
                    note_message = f"{tool_name} 已执行，结果已记录。"

        self.add_relevant_note(note_message)
        if tool_name in {"read_file", "write_file", "patch_file"}:
            path = str(args.get("path", "")).strip()
            if path:
                self.add_recent_file(path)
                self.upsert_file_summary(path, result)

    def retrieve_relevant_notes(self, user_message: str, top_k: int = 5) -> list[str]:
        """
        relevant_notes 已在写入时完成去重与 top_k 裁剪，这里直接返回。
        """
        _ = user_message
        if top_k <= 0:
            return []
        return self.relevant_notes[-top_k:]

    def render_for_prompt(self, user_message: str) -> str:
        """
        组装记忆内容，渲染成提供给提示词的格式
        """
        notes = self.retrieve_relevant_notes(user_message, top_k=TOP_K_RELEVANT_NOTES)
        lines = [
            "## Detail Memory",
            f"- task_summary: {self.task_summary or '(empty)'}",
            f"- recent_files: {', '.join(self.recent_files) if self.recent_files else '(empty)'}",
            "- relevant_notes:",
        ]
        if not notes:
            lines.append("  - (empty)")
        else:
            for n in notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)
