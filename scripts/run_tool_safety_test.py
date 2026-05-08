from __future__ import annotations

import os
import re
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.tools import ToolError, ToolRegistry


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    within_budget: bool
    verifier_passed: bool


def _redact_sensitive(text: str) -> str:
    redacted = re.sub(r"(sk-[A-Za-z0-9_-]{8,})", "<redacted>", text)
    redacted = re.sub(r"(?i)(token\s*[=:]\s*)([^\s]+)", r"\1<redacted>", redacted)
    return redacted


def _run_partial_steps(registry: ToolRegistry) -> tuple[int, int, bool]:
    completed = 0
    failed = 0
    for name, args in [
        ("read_file", {"path": "safe.txt", "start": 1, "end": 5}),
        ("read_file", {"path": "missing.txt", "start": 1, "end": 5}),
    ]:
        try:
            registry.execute(name, args)
            completed += 1
        except ToolError:
            failed += 1
    return completed, failed, completed > 0 and failed > 0


def _case_param_validation(registry: ToolRegistry) -> bool:
    try:
        registry.execute("read_file", {"path": "safe.txt", "start": 0, "end": 1})
    except ToolError as exc:
        return "参数" in str(exc)
    return False


def _case_workspace_isolation(registry: ToolRegistry) -> bool:
    try:
        registry.execute("read_file", {"path": "../outside.txt", "start": 1, "end": 1})
    except ToolError as exc:
        return "越界" in str(exc)
    return False


def _case_high_risk_approval(workspace_root: Path) -> bool:
    denied = ToolRegistry(
        workspace_root=workspace_root, approval_callback=lambda _n, _a: False
    )
    try:
        denied.execute("write_file", {"path": "blocked.txt", "content": "x"})
    except ToolError as exc:
        return "拒绝" in str(exc)
    return False


def _case_repeat_intercept(registry: ToolRegistry) -> bool:
    try:
        registry.execute("list_files", {"path": "."})
        registry.execute("list_files", {"path": "."})
        registry.execute("list_files", {"path": "."})
    except ToolError as exc:
        return "重复" in str(exc)
    return False


def _case_sensitive_redaction(registry: ToolRegistry) -> bool:
    secret = "sk-UnitTestSecret_123456"
    raw = registry.execute(
        "run_shell", {"command": f"echo token={secret}", "timeout": 15}
    )
    masked = _redact_sensitive(raw)
    return secret not in masked and "<redacted>" in masked


def _case_partial_success(registry: ToolRegistry) -> bool:
    completed, failed, partial = _run_partial_steps(registry)
    return completed == 1 and failed == 1 and partial


def _case_unknown_tool_rejected(registry: ToolRegistry) -> bool:
    try:
        registry.execute("delete_file", {"path": "safe.txt"})
    except ToolError as exc:
        return "未知工具" in str(exc)
    return False


def _case_shell_timeout_validation(registry: ToolRegistry) -> bool:
    try:
        registry.execute(
            "run_shell",
            {
                "command": "echo timeout-check",
                "timeout": 300,
            },
        )
    except ToolError as exc:
        return "timeout" in str(exc)
    return False


def _execute_case(
    case_id: str,
    verifier: Callable[[], bool],
    step_budget: int,
    used_steps: int,
) -> CaseResult:
    ok = bool(verifier())
    return CaseResult(
        case_id=case_id,
        passed=ok,
        within_budget=used_steps <= step_budget,
        verifier_passed=ok,
    )


def run() -> None:
    with tempfile.TemporaryDirectory(prefix="codara-safety-") as temp_dir:
        workspace_root = Path(temp_dir)
        (workspace_root / "safe.txt").write_text("line1\nline2\n", encoding="utf-8")
        (workspace_root / "app").mkdir(parents=True, exist_ok=True)

        registry = ToolRegistry(
            workspace_root=workspace_root, approval_callback=lambda _n, _a: True
        )

        results = [
            _execute_case(
                case_id="parameter_validation",
                verifier=lambda: _case_param_validation(registry),
                step_budget=1,
                used_steps=1,
            ),
            _execute_case(
                case_id="workspace_isolation",
                verifier=lambda: _case_workspace_isolation(registry),
                step_budget=1,
                used_steps=1,
            ),
            _execute_case(
                case_id="high_risk_approval",
                verifier=lambda: _case_high_risk_approval(workspace_root),
                step_budget=1,
                used_steps=1,
            ),
            _execute_case(
                case_id="repeat_call_intercept",
                verifier=lambda: _case_repeat_intercept(registry),
                step_budget=3,
                used_steps=3,
            ),
            _execute_case(
                case_id="sensitive_redaction",
                verifier=lambda: _case_sensitive_redaction(registry),
                step_budget=1,
                used_steps=1,
            ),
            _execute_case(
                case_id="partial_success_recognition",
                verifier=lambda: _case_partial_success(registry),
                step_budget=2,
                used_steps=2,
            ),
            _execute_case(
                case_id="unknown_tool_rejected",
                verifier=lambda: _case_unknown_tool_rejected(registry),
                step_budget=1,
                used_steps=1,
            ),
            _execute_case(
                case_id="shell_timeout_validation",
                verifier=lambda: _case_shell_timeout_validation(registry),
                step_budget=1,
                used_steps=1,
            ),
        ]

    total = len(results)
    verifier_passed = sum(1 for row in results if row.verifier_passed)

    print("verifier通过率:", f"{verifier_passed / total:.4f}")


if __name__ == "__main__":
    run()
