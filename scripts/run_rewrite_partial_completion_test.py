from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context_manager import ContextManager
from app.memory import LayeredMemory
from app.models import CompletionResult, DeepSeekChatClient, ModelError
from app.rewrite import RequestRewriter
from app.run_store import RunStore
from app.runtime import CodaraRuntime
from app.tools import ToolRegistry

load_dotenv()


@dataclass
class PartialTask:
    task_id: str
    task_type: str  # "read_write", "search_list", "write_poem", "read_calc", etc.
    user_prompt: str
    verification_keywords: dict[str, bool]  # {keyword: should_contain}


TASKS: list[PartialTask] = [
    PartialTask(
        "t01",
        "read_write",
        "先检索 notfound.py（不存在），再列出 app 目录下的文件。",
        {"失败": True, "notfound.py": True, "已完成": True, "app": True},
    ),
    PartialTask(
        "t02",
        "search_list",
        "搜索关键词 NONEXISTENT_MAGIC，然后列出 tests 目录下所有文件。",
        {"失败": True, "NONEXISTENT_MAGIC": True, "已完成": True, "tests": True},
    ),
    PartialTask(
        "t03",
        "read_compute",
        "读取 missing_file.txt 的内容，然后告诉我 2+2 的结果。",
        {"失败": True, "missing_file.txt": True, "已完成": True, "4": True},
    ),
    PartialTask(
        "t04",
        "write_search",
        "在 app 目录创建文件 temp_test.py，内容为 'test = 1'，然后搜索关键词 'test'。",
        {"已完成": True, "temp_test.py": True, "写入": True},
    ),
    PartialTask(
        "t05",
        "read_sum",
        "打开 ghost_config.ini（不存在），读取配置值，然后计算总和。",
        {"失败": True, "ghost_config.ini": True, "已完成": False},
    ),
    PartialTask(
        "t06",
        "list_filter",
        "列出 docs 目录，找出名字中含 'README' 的文件。",
        {"已完成": True, "docs": True},
    ),
    PartialTask(
        "t07",
        "read_analyze",
        "读取 app/missing_log.json（不存在），分析其中数据结构，然后列出 src 目录。",
        {"失败": True, "missing_log.json": True, "已完成": True, "src": True},
    ),
    PartialTask(
        "t08",
        "write_verify",
        "在 tests 目录新建 verify_test.py，内容为测试代码，然后列出该目录所有文件。",
        {"已完成": True, "verify_test.py": True, "tests": True},
    ),
]


class RewriteModelClient(DeepSeekChatClient):
    """真实 DeepSeek 客户端用于请求重写"""

    pass


class PartialTaskRuntimeModel(DeepSeekChatClient):
    """真实 DeepSeek 客户端用于主循环执行"""

    pass


def _prepare_workspace(root: Path) -> None:
    for dirname in ("app", "docs", "src", "tests"):
        (root / dirname).mkdir(parents=True, exist_ok=True)

    # 添加一些示例文件
    (root / "app" / "sample.py").write_text("# sample\n")
    (root / "docs" / "guide.md").write_text("# Guide\n")
    (root / "src" / "main.py").write_text("# Main\n")
    (root / "tests" / "test_main.py").write_text("# Tests\n")


def _is_partial_completion_success(result: str, task: PartialTask) -> bool:
    """验证任务是否部分成功或完全成功"""
    text = str(result).lower()

    # 检查每个验证关键字
    for keyword, should_contain in task.verification_keywords.items():
        keyword_lower = keyword.lower()
        contains = keyword_lower in text

        if should_contain and not contains:
            # 需要包含但没有
            return False
        elif not should_contain and contains:
            # 不需要包含但有了
            return False

    return True


def _run_single(
    task: PartialTask, use_rewriter: bool, use_explicit_prompt: bool = True
) -> tuple[bool, str]:
    """
    运行单个任务，返回 (成功, 输出结果)

    Args:
        task: 任务对象
        use_rewriter: 是否使用请求重写
        use_explicit_prompt: 是否使用明确的部分完成提示词（True=明确，False=简单）
    """

    # 初始化真实的 DeepSeek API 客户端
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 环境变量未设置。"
            "请在 .env 文件中设置或通过环境变量传入。"
        )

    with tempfile.TemporaryDirectory(prefix="codara-rewrite-partial-") as temp_dir:
        workspace_root = Path(temp_dir)
        _prepare_workspace(workspace_root)

        # 使用真实的 DeepSeek 模型
        main_model = PartialTaskRuntimeModel(
            api_key=api_key,
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            timeout=60,
            temperature=0.2,
        )

        rewriter_model = None
        if use_rewriter:
            rewriter_model = RewriteModelClient(
                api_key=api_key,
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
                timeout=60,
                temperature=0.2,
            )

        rewriter = (
            RequestRewriter(
                model_client=rewriter_model,
                enabled=True,
                max_new_tokens=420,
            )
            if use_rewriter
            else None
        )

        # 自定义 ContextManager 的提示词版本
        class CustomContextManager(ContextManager):
            def _assemble(self, render_map, user_message: str) -> str:
                if use_explicit_prompt:
                    # 明确的提示词：强调部分完成和多子任务处理
                    return (
                        "你是 Codara。你必须严格按链路执行，不得跳过或循环。\n"
                        "当前输入已由请求重写器整理为'用户请求 + 子任务'结构。\n"
                        "你必须优先按子任务序号推进，逐项执行并记录结果。\n"
                        "【输出协议】\n"
                        "- 每轮只允许输出一种标签：<tool>...</tool> 或 <final>...</final>。\n"
                        '- 需要调用工具时，唯一合法格式：<tool>{"name":"工具名","args":{...}}</tool>\n'
                        "- 能直接回答时，输出 <final>你的答案</final>。\n"
                        "- 当请求含多个子任务时，允许部分完成；必须在 <final> 中说明已完成项与失败项。\n"
                        "- 生成 <final> 时，按子任务编号输出每项状态（完成/失败）与简要依据。\n"
                        "\n【执行链路】\n"
                        "1) 先解析'用户请求/子任务'并锁定当前要执行的子任务编号。\n"
                        "2) 仅在必要时做最小化探测（list_files/read_file）。\n"
                        "3) 形成可执行动作后立即执行写入/补丁工具，不得反复探测同一信息。\n"
                        "4) 若某子任务报错但其余可继续，继续执行其余子任务。\n"
                        "5) 工具结果满足目标后立刻 <final>；若部分失败，也应输出高质量分项总结。\n"
                        "\n【硬性约束】\n"
                        "- 每次只能调用一个工具。\n"
                        "- 严禁连续重复同一工具+同一参数。\n"
                        "- 禁止为了确认已确认的信息而重复 list_files/read_file。\n"
                        "- 高风险工具谨慎执行，遵守审批。\n"
                        "- 创建目录/文件优先 write_file（自动创建父目录），仅必要时 run_shell。\n"
                        "\n可用工具：\n"
                        f"{render_map['workspace'].rendered}\n\n"
                        f"{render_map['detail_memory'].rendered}\n\n"
                        f"{render_map['general_memory'].rendered}\n\n"
                        f"{user_message.strip()}\n"
                    )
                else:
                    # 简单的提示词：不强调部分完成
                    return (
                        "你是 Codara，一个代码助手。\n"
                        "请执行用户的请求。\n"
                        "【输出协议】\n"
                        "- 输出只能是 <tool>...</tool> 或 <final>...</final>。\n"
                        '- 调用工具时：<tool>{"name":"工具名","args":{...}}</tool>\n'
                        "- 完成时：<final>你的答案</final>\n"
                        "\n可用工具：\n"
                        f"{render_map['workspace'].rendered}\n\n"
                        f"{user_message.strip()}\n"
                    )

        runtime = CodaraRuntime(
            workspace_root=workspace_root,
            model_client=main_model,
            tool_registry=ToolRegistry(
                workspace_root=workspace_root, approval_callback=lambda _n, _a: True
            ),
            memory=LayeredMemory(),
            run_store=RunStore(workspace_root=workspace_root),
            context_manager=CustomContextManager(max_total_chars=5000),
            request_rewriter=rewriter,
            max_steps=8,
            max_retries=3,
        )

        answer = runtime.ask(task.user_prompt)
        success = _is_partial_completion_success(answer, task)
        return success, answer


def run() -> None:
    if len(TASKS) != 8:
        raise ValueError(f"expected 8 tasks, got {len(TASKS)}")

    # 检查 API Key
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("错误: DEEPSEEK_API_KEY 环境变量未设置。")
        print("请先在 .env 文件中设置或通过环境变量传入:")
        print("  export DEEPSEEK_API_KEY=your-api-key")
        return

    print("=" * 90)
    print("部分完成率对比测试：Baseline vs Improved")
    print("=" * 90)
    print()
    print("Baseline (基础)：无重写 + 简单提示词（不强调部分完成）")
    print("Improved (改进)：有重写 + 明确提示词（强调部分完成和多子任务）")
    print()

    baseline_success = 0
    improved_success = 0

    for idx, task in enumerate(TASKS, start=1):
        print(f"【任务 {task.task_id}】 ({idx}/8)")
        print(f"输入问题：{task.user_prompt}")
        print()

        try:
            # Baseline: 无重写 + 简单提示词
            print("  [Baseline]")
            baseline_ok, baseline_result = _run_single(
                task, use_rewriter=False, use_explicit_prompt=False
            )
            result_preview = baseline_result.replace("\n", " ")[:100]
            print(f"    {result_preview}...")
            if baseline_ok:
                baseline_success += 1

            print()

            # Improved: 有重写 + 明确提示词
            print("  [Improved]")
            improved_ok, improved_result = _run_single(
                task, use_rewriter=True, use_explicit_prompt=True
            )
            result_preview = improved_result.replace("\n", " ")[:100]
            print(f"    {result_preview}...")
            if improved_ok:
                improved_success += 1

            print()

        except ModelError as e:
            print(f"    [错误] {e}")
            print()
        except Exception as e:
            print(f"    [错误] {e}")
            print()

    print("=" * 90)
    print("测试结果汇总")
    print("=" * 90)
    total = len(TASKS)
    baseline_rate = baseline_success / total if total > 0 else 0
    improved_rate = improved_success / total if total > 0 else 0
    improvement = (improved_rate - baseline_rate) * 100

    print(f"总测试组数：{total}")
    print()
    print(f"[Baseline] 无重写 + 简单提示词")
    print(f"  完成数：{baseline_success}/{total}")
    print(f"  成功率：{baseline_rate:.2%}")
    print()
    print(f"[Improved] 有重写 + 明确提示词")
    print(f"  完成数：{improved_success}/{total}")
    print(f"  成功率：{improved_rate:.2%}")
    print()
    print(
        f"改进幅度：{improvement:+.2f}% (从 {baseline_rate:.2%} 提升到 {improved_rate:.2%})"
    )
    print("=" * 90)


if __name__ == "__main__":
    run()
