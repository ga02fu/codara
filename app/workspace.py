from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


# 白名单文档
DOC_WHITELIST = [
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "package.json",
]


def _run_git(root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except Exception:
        return ""


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n...[truncated]"


@dataclass
class WorkspaceContext:
    """
    代码仓库上下文快照。
    给模型提供一个便宜但有方向感的基线，而不是预加载全仓。
    启动后采集 cwd repo_root branch status recent_commits 等参数
    再补充白名单文档摘要，渲染稳定的 workspace 文本，后续供 prompt 使用
    """

    cwd: str
    repo_root: str
    branch: str
    default_branch: str
    status: str
    recent_commits: str
    project_docs: dict[str, str]

    @classmethod
    def build(cls, root: Path) -> "WorkspaceContext":
        """
        构建工作区上下文
        """
        root = root.resolve()
        repo_root = _run_git(root, ["rev-parse", "--show-toplevel"]) or str(root)
        branch = _run_git(root, ["branch", "--show-current"])
        default_branch = _run_git(
            root, ["symbolic-ref", "refs/remotes/origin/HEAD"]
        ).split("/")[-1]
        status = _truncate(_run_git(root, ["status", "--short"]), 1500)
        recent_commits = _truncate(
            _run_git(root, ["log", "--oneline", "-n", "5"]), 1200
        )

        # 扫描白名单文档
        docs: dict[str, str] = {}
        for rel in DOC_WHITELIST:
            p = root / rel
            if p.exists() and p.is_file():
                docs[rel] = _truncate(
                    p.read_text(encoding="utf-8", errors="ignore"), 1200
                )

        return cls(
            cwd=str(root),
            repo_root=repo_root,
            branch=branch,
            default_branch=default_branch,
            status=status,
            recent_commits=recent_commits,
            project_docs=docs,
        )

    def render(self) -> str:
        """
        渲染工作区相关上下文文本
        """
        lines = [
            "## Workspace Snapshot",
            f"- cwd: {self.cwd}",
            f"- repo_root: {self.repo_root}",
            f"- branch: {self.branch or 'N/A'}",
            f"- default_branch: {self.default_branch or 'N/A'}",
            "",
            "### git status (short)",
            self.status or "(empty)",
            "",
            "### recent commits",
            self.recent_commits or "(empty)",
            "",
            "### project docs (whitelist)",
        ]
        if not self.project_docs:
            lines.append("(none)")
        else:
            for name, content in self.project_docs.items():
                lines.append(f"\n#### {name}\n{content}")
        return "\n".join(lines)
