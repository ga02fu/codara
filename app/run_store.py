from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunStore:
    """
    会话落盘存储。
    使用本地 JSON，路径固定在 .codara/sessions 下
    """

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.base_dir = workspace_root / ".codara" / "sessions"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.json"

    def save(self, session_id: str, payload: dict[str, Any]) -> None:
        path = self.path_for(session_id)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self.path_for(session_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def latest_session_id(self) -> str | None:
        files = sorted(self.base_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return None
        return files[0].stem