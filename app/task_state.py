from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskState:
    """
    单轮任务状态。
    每轮执行都可追踪，可解释停止原因。
    """

    run_id: str
    task_id: str
    status: str = "running"
    attempts: int = 0
    tool_steps: int = 0
    stop_reason: str | None = None
    started_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
    error: str | None = None

    def finish(self, reason: str) -> None:
        self.status = "completed"
        self.stop_reason = reason
        self.finished_at = now_iso()

    def stop(self, reason: str) -> None:
        self.status = "stopped"
        self.stop_reason = reason
        self.finished_at = now_iso()

    def fail(self, error: str) -> None:
        self.status = "failed"
        self.error = error
        self.finished_at = now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "status": self.status,
            "attempts": self.attempts,
            "tool_steps": self.tool_steps,
            "stop_reason": self.stop_reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }
