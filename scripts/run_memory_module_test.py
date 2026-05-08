from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context_manager import ContextManager
from app.memory import LayeredMemory
from app.models import CompletionResult
from app.run_store import RunStore
from app.runtime import CodaraRuntime
from app.tools import ToolRegistry


@dataclass
class MemoryTask:
    task_id: str
    task_type: str
    fact_line: str
    files_in_app: list[str]
    marker_symbol: str
    marker_file: str
    write_target: str | None = None


TASK_BLUEPRINTS: list[tuple[str, str, list[str], str, str]] = [
    (
        "fact_lookup",
        "deploy key is red",
        ["deploy.py", "keys.py"],
        "DEPLOY_RED",
        "app/deploy.py",
    ),
    (
        "edit_dependency",
        "fixed field name is benchmark_schema",
        ["config.py", "schema.py", "loader.py"],
        "SCHEMA_LOCKED",
        "app/schema.py",
    ),
    (
        "debug_reference",
        "retry ceiling is 2",
        ["runtime.py", "retry.py", "errors.py"],
        "RETRY_CEILING_2",
        "app/retry.py",
    ),
    (
        "workflow_trace",
        "inspection sequence is read_file then list_files",
        ["trace.py", "steps.py", "planner.py"],
        "FLOW_READ_THEN_LIST",
        "app/trace.py",
    ),
]


def _build_tasks() -> list[MemoryTask]:
    tasks: list[MemoryTask] = []
    for i in range(4):
        for j, (task_type, fact, files, marker, marker_file) in enumerate(
            TASK_BLUEPRINTS, start=1
        ):
            task_idx = i * len(TASK_BLUEPRINTS) + j
            tasks.append(
                MemoryTask(
                    task_id=f"t{task_idx:02d}",
                    task_type=task_type,
                    fact_line=f"{fact} [case-{task_idx}]",
                    files_in_app=[f"case_{task_idx}_{name}" for name in files],
                    marker_symbol=f"{marker}_CASE_{task_idx}",
                    marker_file=marker_file.replace(".py", f"_{task_idx}.py"),
                )
            )

    # 新增 4 组多步骤任务：先 read/search，再 write_file 到 docs 目录
    for idx in range(17, 21):
        tasks.append(
            MemoryTask(
                task_id=f"t{idx:02d}",
                task_type="multi_step_read_write",
                fact_line=f"document key fact for case-{idx}",
                files_in_app=[
                    f"case_{idx}_source.py",
                    f"case_{idx}_notes.py",
                    f"case_{idx}_helper.py",
                ],
                marker_symbol=f"DOC_WRITE_FLOW_CASE_{idx}",
                marker_file=f"app/write_flow_{idx}.py",
                write_target=f"docs/generated_case_{idx}.md",
            )
        )

    if len(tasks) != 20:
        raise ValueError(f"expected 20 tasks, got {len(tasks)}")
    return tasks


TASKS = _build_tasks()


class ScriptedMemoryModel:
    def __init__(
        self,
        task_type: str,
        expected_fact: str,
        expected_files: list[str],
        marker_symbol: str,
        write_target: str | None,
    ):
        self.task_type = task_type
        self.expected_fact = expected_fact
        self.expected_files = [item.lower() for item in expected_files]
        self.marker_symbol = marker_symbol.lower()
        self.write_target = str(write_target or "").strip()
        self.phase = "bootstrap_read"
        self.followup_read_done = False
        self.followup_list_done = False
        self.followup_search_done = False
        self.followup_write_done = False

    @staticmethod
    def _build_summary(tool_name: str, result: str, args_text: str) -> str:
        compact = str(result).replace("\n", " | ")
        compact = " ".join(compact.split())
        if tool_name == "read_file":
            return f"read_file 已提取关键事实: {compact[:160]}"
        if tool_name == "list_files":
            return f"list_files 已确认目录文件: {compact[:160]}"
        if tool_name == "search":
            return f"search 已定位标记: {compact[:160]}"
        if tool_name == "write_file":
            return f"write_file 已写入文档: {compact[:160]}"
        return f"{tool_name} 已执行: {args_text[:80]}"

    def complete(self, prompt: str, max_new_tokens: int = 1200) -> CompletionResult:
        del max_new_tokens
        prompt_lower = prompt.lower()

        if "你是记忆摘要器" in prompt:
            tool_name = ""
            args_text = ""
            result_text = ""
            for line in prompt.splitlines():
                if line.startswith("tool_name="):
                    tool_name = line.split("=", 1)[1].strip()
                elif line.startswith("args="):
                    args_text = line.split("=", 1)[1].strip()
                elif line.startswith("result="):
                    result_text = line.split("=", 1)[1].strip()
            return CompletionResult(
                text=json.dumps(
                    {"summary": self._build_summary(tool_name, result_text, args_text)},
                    ensure_ascii=False,
                ),
                usage={},
                raw={},
            )

        if "=== user request ===" in prompt_lower:
            user_request = prompt_lower.split("=== user request ===", 1)[1]
        else:
            user_request = prompt_lower

        if "预热任务" in user_request:
            if self.phase == "bootstrap_read":
                self.phase = "bootstrap_list"
                return CompletionResult(
                    text='<tool>{"name":"read_file","args":{"path":"facts.txt","start":1,"end":20}}</tool>',
                    usage={},
                    raw={},
                )

            if self.phase == "bootstrap_list":
                if self.task_type == "fact_lookup":
                    self.phase = "bootstrap_final"
                    return CompletionResult(
                        text="<final>bootstrap done</final>", usage={}, raw={}
                    )
                self.phase = "bootstrap_search"
                return CompletionResult(
                    text='<tool>{"name":"list_files","args":{"path":"app"}}</tool>',
                    usage={},
                    raw={},
                )

            if self.phase == "bootstrap_search":
                self.phase = "bootstrap_final"
                return CompletionResult(
                    text=f'<tool>{{"name":"search","args":{{"pattern":"{self.marker_symbol}","path":"app"}}}}</tool>',
                    usage={},
                    raw={},
                )

            return CompletionResult(
                text="<final>bootstrap done</final>", usage={}, raw={}
            )

        general_text = ""
        detail_text = ""
        if (
            "=== detail memory ===" in prompt_lower
            and "=== general memory ===" in prompt_lower
        ):
            detail_text = prompt_lower.split("=== detail memory ===", 1)[1].split(
                "=== general memory ===", 1
            )[0]
        if (
            "=== general memory ===" in prompt_lower
            and "=== user request ===" in prompt_lower
        ):
            general_text = prompt_lower.split("=== general memory ===", 1)[1].split(
                "=== user request ===", 1
            )[0]

        memory_view = (detail_text + "\n" + general_text).lower()
        has_fact = self.expected_fact.lower() in memory_view
        has_file_signal = bool(self.expected_files) and all(
            name in memory_view for name in self.expected_files
        )
        has_marker_signal = self.marker_symbol in memory_view

        if (
            self.task_type
            in {
                "edit_dependency",
                "debug_reference",
                "workflow_trace",
                "multi_step_read_write",
            }
            and not has_file_signal
            and "目录已检查" in general_text
        ):
            has_file_signal = True

        needs_marker = self.task_type in {
            "debug_reference",
            "workflow_trace",
            "multi_step_read_write",
        }
        if has_fact and has_file_signal and (has_marker_signal or not needs_marker):
            if (
                self.task_type == "multi_step_read_write"
                and self.write_target
                and not self.followup_write_done
            ):
                self.followup_write_done = True
                content = (
                    f"# generated report\n"
                    f"task_type: {self.task_type}\n"
                    f"fact: {self.expected_fact}\n"
                    f"marker: {self.marker_symbol}\n"
                )
                payload = {
                    "name": "write_file",
                    "args": {"path": self.write_target, "content": content},
                }
                return CompletionResult(
                    text=f"<tool>{json.dumps(payload, ensure_ascii=False)}</tool>",
                    usage={},
                    raw={},
                )
            return CompletionResult(
                text="<final>memory answer ready</final>", usage={}, raw={}
            )

        if not has_fact and not self.followup_read_done:
            self.followup_read_done = True
            return CompletionResult(
                text='<tool>{"name":"read_file","args":{"path":"facts.txt","start":1,"end":20}}</tool>',
                usage={},
                raw={},
            )

        if not has_file_signal and not self.followup_list_done:
            self.followup_list_done = True
            return CompletionResult(
                text='<tool>{"name":"list_files","args":{"path":"app"}}</tool>',
                usage={},
                raw={},
            )

        if needs_marker and not has_marker_signal and not self.followup_search_done:
            self.followup_search_done = True
            return CompletionResult(
                text=f'<tool>{{"name":"search","args":{{"pattern":"{self.marker_symbol}","path":"app"}}}}</tool>',
                usage={},
                raw={},
            )

        return CompletionResult(
            text="<final>memory answer ready</final>", usage={}, raw={}
        )


def _build_runtime(
    workspace_root: Path,
    task_type: str,
    expected_fact: str,
    expected_files: list[str],
    marker_symbol: str,
    write_target: str | None,
) -> CodaraRuntime:
    return CodaraRuntime(
        workspace_root=workspace_root,
        model_client=ScriptedMemoryModel(
            task_type,
            expected_fact,
            expected_files,
            marker_symbol,
            write_target,
        ),
        tool_registry=ToolRegistry(
            workspace_root=workspace_root, approval_callback=lambda _n, _a: True
        ),
        memory=LayeredMemory(),
        run_store=RunStore(workspace_root=workspace_root),
        context_manager=ContextManager(max_total_chars=5000),
        request_rewriter=None,
        max_steps=12,
        max_retries=2,
    )


def _apply_variant(runtime: CodaraRuntime, variant: str) -> None:
    if variant == "no_memory":
        runtime.general_memory = []
        runtime.memory = LayeredMemory()
        runtime.memory.summarize_tool_fn = runtime._llm_summarize_tool_note
        return

    if variant == "general_only":
        compact_general: list[dict[str, str]] = []
        for item in runtime.general_memory:
            role = str(item.get("role", "")).strip().lower()
            if role == "tool":
                continue
            compact_general.append(item)
        compact_general.append(
            {
                "role": "assistant",
                "content": "预热已完成：目录已检查，可直接回答高层问题。",
            }
        )
        runtime.general_memory = compact_general[-20:]
        runtime.memory = LayeredMemory()
        runtime.memory.summarize_tool_fn = runtime._llm_summarize_tool_note


def _persist_written_doc(
    workspace_root: Path,
    task: MemoryTask,
    variant: str,
) -> None:
    if not task.write_target:
        return
    source = workspace_root / task.write_target
    if not source.exists():
        return

    docs_root = ROOT / "docs" / "memory_module_outputs"
    docs_root.mkdir(parents=True, exist_ok=True)
    target = docs_root / f"{task.task_id}_{variant}.md"
    content = source.read_text(encoding="utf-8", errors="ignore")
    decorated = (
        f"# persisted output\n"
        f"task_id: {task.task_id}\n"
        f"variant: {variant}\n"
        f"task_type: {task.task_type}\n\n"
        f"{content}"
    )
    target.write_text(decorated, encoding="utf-8")


def _run_single_task(task: MemoryTask, variant: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="codara-memory-") as temp_dir:
        workspace_root = Path(temp_dir)
        (workspace_root / "app").mkdir(parents=True, exist_ok=True)

        for name in task.files_in_app:
            (workspace_root / "app" / name).write_text(
                "placeholder\n", encoding="utf-8"
            )

        marker_path = workspace_root / task.marker_file
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(f"MARKER={task.marker_symbol}\n", encoding="utf-8")

        if task.write_target:
            (workspace_root / "docs").mkdir(parents=True, exist_ok=True)

        (workspace_root / "facts.txt").write_text(
            task.fact_line + "\n", encoding="utf-8"
        )

        runtime = _build_runtime(
            workspace_root=workspace_root,
            task_type=task.task_type,
            expected_fact=task.fact_line,
            expected_files=task.files_in_app,
            marker_symbol=task.marker_symbol,
            write_target=task.write_target,
        )

        runtime.ask("预热任务：读取 facts.txt，检查 app 目录，并定位标记。")
        _apply_variant(runtime, variant)

        before = len(runtime.trace)
        if task.task_type == "multi_step_read_write":
            runtime.ask(
                "追问任务：先读取关键事实并确认标记，再将结论写入 docs 目录中的报告文件。"
            )
        else:
            runtime.ask(
                "追问任务：在不重复探测的前提下，回答关键事实并说明相关文件与标记。"
            )
        after_trace = runtime.trace[before:]

        followup_tool_steps = sum(
            1 for event in after_trace if event.get("type") == "tool_result"
        )

        _persist_written_doc(workspace_root, task, variant)

        return {"tool_steps": followup_tool_steps}


def _calc_stats(rows: list[dict[str, int]]) -> dict[str, float]:
    steps = [row["tool_steps"] for row in rows]
    return {"avg_tool_steps": (sum(steps) / len(steps)) if steps else 0.0}


def _print_group_report(
    title: str,
    tasks: list[MemoryTask],
    rows: dict[str, list[dict[str, int]]],
) -> None:
    print(f"\n[{title}] 任务数: {len(tasks)}")
    for variant in ("no_memory", "general_only", "general_plus_detail"):
        stats = _calc_stats(rows[variant])
        print(f"{variant}: 平均工具步数={stats['avg_tool_steps']:.2f}")


def run() -> None:
    variants = ["no_memory", "general_only", "general_plus_detail"]
    overall_rows: dict[str, list[dict[str, int]]] = {name: [] for name in variants}

    groups: list[tuple[str, list[MemoryTask]]] = [
        (
            "类型=fact_lookup",
            [task for task in TASKS if task.task_type == "fact_lookup"],
        ),
        (
            "类型=edit_dependency",
            [task for task in TASKS if task.task_type == "edit_dependency"],
        ),
        (
            "类型=debug_reference",
            [task for task in TASKS if task.task_type == "debug_reference"],
        ),
        (
            "类型=workflow_trace",
            [task for task in TASKS if task.task_type == "workflow_trace"],
        ),
        (
            "类型=multi_step_read_write",
            [task for task in TASKS if task.task_type == "multi_step_read_write"],
        ),
    ]

    print("记忆模块工具效率测试")
    for title, group_tasks in groups:
        group_rows: dict[str, list[dict[str, int]]] = {name: [] for name in variants}
        for task in group_tasks:
            no_memory_row = _run_single_task(task, "no_memory")
            general_only_row = _run_single_task(task, "general_only")
            full_memory_row = _run_single_task(task, "general_plus_detail")

            group_rows["no_memory"].append(no_memory_row)
            group_rows["general_only"].append(general_only_row)
            group_rows["general_plus_detail"].append(full_memory_row)

            overall_rows["no_memory"].append(no_memory_row)
            overall_rows["general_only"].append(general_only_row)
            overall_rows["general_plus_detail"].append(full_memory_row)

        _print_group_report(title, group_tasks, group_rows)

    print(f"\n[总览] 任务覆盖数: {len(TASKS)}")
    for variant in variants:
        stats = _calc_stats(overall_rows[variant])
        print(f"{variant}: 平均工具步数={stats['avg_tool_steps']:.2f}")
    print("\n落盘目录: docs/memory_module_outputs")


if __name__ == "__main__":
    run()
