from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class ToolError(Exception):
    pass


ApprovalCallback = Callable[[str, dict[str, Any]], bool]


@dataclass
class ToolSpec:
    name: str
    risky: bool
    run: Callable[[dict[str, Any]], str]


class ToolRegistry:
    """
    工具注册与执行中心。
    显式白名单 + 参数校验 + 路径边界 + 风险审批。
    是一个安全的受控执行链
    """

    def __init__(
        self, workspace_root: Path, approval_callback: ApprovalCallback | None = None
    ):
        self.workspace_root = workspace_root.resolve()
        self.approval_callback = approval_callback
        self._last_call_signature = ""
        self._repeat_count = 0
        self.tools: dict[str, ToolSpec] = {}
        self._register_builtin_tools()

    def reset_round_state(self) -> None:
        """
        每轮用户请求前重置去重状态，避免跨轮误拦截。
        """
        self._last_call_signature = ""
        self._repeat_count = 0

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": spec.name, "risky": spec.risky} for spec in self.tools.values()
        ]

    def execute(self, name: str, args: dict[str, Any]) -> str:
        if name not in self.tools:
            raise ToolError(f"未知工具: {name}")

        spec = self.tools[name]
        sig = json.dumps(
            {"name": name, "args": args}, ensure_ascii=False, sort_keys=True
        )
        if sig == self._last_call_signature:
            self._repeat_count += 1
        else:
            self._last_call_signature = sig
            self._repeat_count = 1

        # 连续相同调用最多允许 2 次，第 3 次拦截。
        repeat_limit = 3
        if self._repeat_count >= repeat_limit:
            raise ToolError("检测到连续重复工具调用，已拦截。")

        if spec.risky and self.approval_callback:
            approved = self.approval_callback(name, args)
            if not approved:
                raise ToolError(f"工具 {name} 被用户拒绝执行。")

        return spec.run(args)

    def _register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def _resolve_in_workspace(self, path_value: str) -> Path:
        if not path_value:
            raise ToolError("path 不能为空。")
        # 支持相对路径和绝对路径两种形式：若传入绝对路径，直接解析；
        # 若传入相对路径，则基于 workspace_root 解析。
        p_val = str(path_value)
        if Path(p_val).is_absolute():
            candidate = Path(p_val).resolve()
        else:
            candidate = (self.workspace_root / p_val).resolve()
        root = self.workspace_root.resolve()

        # 更稳定的边界判断
        try:
            common = os.path.commonpath([str(root), str(candidate)])
            if common != str(root):
                raise ToolError(f"路径越界: {path_value}")
        except Exception:
            if not str(candidate).startswith(str(root)):
                raise ToolError(f"路径越界: {path_value}")

        return candidate

    def _register_builtin_tools(self) -> None:
        """
        注册内置工具
        """
        self._register(ToolSpec("list_files", False, self._tool_list_files))
        self._register(ToolSpec("read_file", False, self._tool_read_file))
        self._register(ToolSpec("search", False, self._tool_search))
        self._register(ToolSpec("write_file", True, self._tool_write_file))
        self._register(ToolSpec("patch_file", True, self._tool_patch_file))
        self._register(ToolSpec("run_shell", True, self._tool_run_shell))

    def _tool_list_files(self, args: dict[str, Any]) -> str:
        rel = str(args.get("path", "."))
        p = self._resolve_in_workspace(rel)
        if not p.exists():
            raise ToolError(f"目录不存在: {rel}")
        if not p.is_dir():
            raise ToolError(f"不是目录: {rel}")
        items = sorted([x.name + ("/" if x.is_dir() else "") for x in p.iterdir()])
        return "\n".join(items[:300])

    def _tool_read_file(self, args: dict[str, Any]) -> str:
        rel = str(args.get("path", "")).strip()
        start = int(args.get("start", 1))
        end = int(args.get("end", start + 200))
        if start <= 0 or end < start:
            raise ToolError("start/end 参数不合法。")

        p = self._resolve_in_workspace(rel)
        if not p.exists() or not p.is_file():
            raise ToolError(f"文件不存在: {rel}")

        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        sliced = lines[start - 1 : end]
        numbered = [f"{i}:{line}" for i, line in enumerate(sliced, start=start)]
        return "\n".join(numbered)

    def _tool_search(self, args: dict[str, Any]) -> str:
        pattern = str(args.get("pattern", "")).strip()
        rel = str(args.get("path", ".")).strip()
        if not pattern:
            raise ToolError("pattern 不能为空。")

        root = self._resolve_in_workspace(rel)

        # 优先使用 rg，速度更快
        if shutil.which("rg"):
            cmd = ["rg", "-n", "-S", pattern, str(root)]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=20, check=False
            )
            output = proc.stdout.strip()
            return output[:8000] if output else "(no matches)"

        # 无 rg 时使用 python 正则兜底
        rx = re.compile(pattern, re.IGNORECASE)
        hits: list[str] = []
        for fp in root.rglob("*"):
            if not fp.is_file():
                continue
            try:
                for idx, line in enumerate(
                    fp.read_text(encoding="utf-8", errors="ignore").splitlines(),
                    start=1,
                ):
                    if rx.search(line):
                        hits.append(
                            f"{fp.relative_to(self.workspace_root)}:{idx}:{line}"
                        )
                        if len(hits) >= 200:
                            break
            except Exception:
                continue
            if len(hits) >= 200:
                break
        return "\n".join(hits) if hits else "(no matches)"

    def _tool_write_file(self, args: dict[str, Any]) -> str:
        rel = str(args.get("path", "")).strip()
        content = str(args.get("content", ""))
        p = self._resolve_in_workspace(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"写入成功: {rel} ({len(content)} chars)"

    def _tool_patch_file(self, args: dict[str, Any]) -> str:
        rel = str(args.get("path", "")).strip()
        old_text = str(args.get("old_text", ""))
        new_text = str(args.get("new_text", ""))
        if not old_text:
            raise ToolError("old_text 不能为空。")

        p = self._resolve_in_workspace(rel)
        if not p.exists() or not p.is_file():
            raise ToolError(f"文件不存在: {rel}")

        text = p.read_text(encoding="utf-8", errors="ignore")
        count = text.count(old_text)
        if count == 0:
            raise ToolError("old_text 未匹配到。")
        if count > 1:
            raise ToolError("old_text 匹配到多处，MVP 仅允许精确唯一替换。")

        patched = text.replace(old_text, new_text, 1)
        p.write_text(patched, encoding="utf-8")
        return f"补丁成功: {rel}"

    def _tool_run_shell(self, args: dict[str, Any]) -> str:
        # 兼容模型偶发输出 cmd 字段，降低因参数名漂移导致的失败率。
        command = str(args.get("command") or args.get("cmd") or "").strip()
        timeout = int(args.get("timeout", 20))
        if not command:
            raise ToolError("command 不能为空。")
        if timeout <= 0 or timeout > 120:
            raise ToolError("timeout 需在 1-120 秒之间。")

        # Windows 下将 mkdir 参数中的 / 归一化为 \\，减少命令语法错误。
        if os.name == "nt" and command.lower().startswith("mkdir "):
            prefix, path_part = command.split(" ", 1)
            command = f"{prefix} {path_part.replace('/', '\\')}"

        proc = subprocess.run(
            command,
            cwd=self.workspace_root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        return (
            f"[exit_code] {proc.returncode}\n"
            f"[stdout]\n{out[:4000]}\n"
            f"[stderr]\n{err[:2000]}"
        )
