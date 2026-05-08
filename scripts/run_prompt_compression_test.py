from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context_manager import ContextManager
from app.memory import LayeredMemory


def _load_real_prompts() -> list[str]:
    benchmark_path = ROOT.parent / "pico-main" / "benchmarks" / "coding_tasks.json"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"missing benchmark file: {benchmark_path}")
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark_prompts = [
        str(task.get("prompt", "")).strip() for task in payload.get("tasks", [])
    ]
    benchmark_prompts = [item for item in benchmark_prompts if item]
    if not benchmark_prompts:
        raise ValueError("benchmark prompts are empty")

    codara_extra_prompts = [
        "Inspect app/runtime.py and explain when duplicate_tool_loop is triggered.",
        "Read app/tools.py and summarize workspace boundary checks for read_file.",
        "Compare app/memory.py relevant note dedup logic with recent file tracking behavior.",
        "Open app/context_manager.py and describe section reduction order under overflow.",
        "Review app/cli.py approval mode switching path and list command-side effects.",
        "Inspect app/workspace.py and explain git metadata fallback behavior without repository.",
        "Read app/run_store.py and outline session save/load file layout.",
        "Inspect app/task_state.py and summarize stop_reason/status transitions.",
        "Open app/rewrite.py and explain how rewritten requests are injected into runtime flow.",
        "Read pyproject.toml and summarize executable entry points and runtime dependencies.",
    ]

    prompts = [*benchmark_prompts, *codara_extra_prompts]
    unique_prompts: list[str] = []
    seen: set[str] = set()
    for item in prompts:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_prompts.append(normalized)

    target_count = 16
    if len(unique_prompts) < target_count:
        raise ValueError(
            f"insufficient unique prompts: expected {target_count}, got {len(unique_prompts)}"
        )
    prompts = unique_prompts[:target_count]

    if len(set(prompts)) != target_count:
        raise ValueError("prompt inputs are not unique")
    return prompts


def _build_real_context_inputs() -> tuple[str, str, list[dict[str, str]]]:
    runtime_text = (ROOT / "app" / "runtime.py").read_text(encoding="utf-8")
    tools_text = (ROOT / "app" / "tools.py").read_text(encoding="utf-8")
    workspace_text = (
        "workspace overview:\n"
        + "- app/runtime.py excerpt\n"
        + runtime_text[:2400]
        + "\n\n- app/tools.py excerpt\n"
        + tools_text[:2400]
    )

    memory = LayeredMemory()
    memory.set_task_summary("benchmark prompts compression check")
    memory.add_recent_file("app/runtime.py")
    memory.add_recent_file("app/tools.py")
    memory.add_relevant_note(
        "read_file app/runtime.py completed; control loop confirmed"
    )
    memory.add_relevant_note(
        "read_file app/tools.py completed; safety guards confirmed"
    )
    detail_memory_text = memory.render_for_prompt("compression baseline")

    general_memory = [
        {
            "role": "user",
            "content": "Summarize the runtime behavior before editing any file.",
        },
        {
            "role": "assistant",
            "content": "I will inspect runtime.py and tools.py, then report safety constraints.",
        },
        {
            "role": "tool",
            "content": "read_file(app/runtime.py) -> control loop, retries, and stop reasons found.",
        },
        {
            "role": "tool",
            "content": "read_file(app/tools.py) -> workspace boundary and repeat-call intercept found.",
        },
    ]
    return workspace_text, detail_memory_text, general_memory


def _case_seed(prompt: str) -> int:
    digest = hashlib.sha1(prompt.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:12], 16)


def _build_case_context_inputs(
    prompt: str,
    base_workspace_text: str,
    base_detail_memory_text: str,
    base_general_memory: list[dict[str, str]],
) -> tuple[str, str, list[dict[str, str]]]:
    seed = _case_seed(prompt)

    workspace_keep = 700 + (seed % 3600)
    detail_keep = 140 + (seed % 700)
    general_keep = 2 + (seed % 3)

    workspace_text = base_workspace_text[:workspace_keep]
    detail_memory_text = base_detail_memory_text[:detail_keep] + (
        f"\n- case_hint: prompt_seed_{seed % 10000}"
    )

    general_memory = [dict(item) for item in base_general_memory[:general_keep]]
    general_memory.append(
        {
            "role": "user",
            "content": f"case-{seed % 10000}: {prompt[:90]}",
        }
    )
    general_memory.append(
        {
            "role": "assistant",
            "content": "Use minimal probing and preserve recent context details.",
        }
    )
    return workspace_text, detail_memory_text, general_memory


def _build_uncompressed_manager(
    system_prefix: str,
    workspace_text: str,
    detail_memory_text: str,
    general_memory: list[dict[str, str]],
) -> ContextManager:
    general_raw = "\n".join(
        f"[{row.get('role', 'unknown')}] {row.get('content', '')}"
        for row in general_memory
    )
    budgets = {
        "system_prefix": len(system_prefix) + 256,
        "workspace": len(workspace_text) + 256,
        "detail_memory": len(detail_memory_text) + 256,
        "general_memory": len(general_raw) + 256,
    }
    return ContextManager(max_total_chars=200000, section_budgets=budgets)


def run() -> None:
    prompts = _load_real_prompts()
    base_workspace_text, base_detail_memory_text, base_general_memory = (
        _build_real_context_inputs()
    )

    system_prefix = (
        "You are Codara. Follow one-tool-per-step policy and avoid repeated probing. "
        "Prefer direct edits after minimal inspection."
    )

    original_chars: list[int] = []
    compressed_chars: list[int] = []

    for prompt in prompts:
        workspace_text, detail_memory_text, general_memory = _build_case_context_inputs(
            prompt=prompt,
            base_workspace_text=base_workspace_text,
            base_detail_memory_text=base_detail_memory_text,
            base_general_memory=base_general_memory,
        )
        compressed_manager = ContextManager(max_total_chars=3600)
        uncompressed_manager = _build_uncompressed_manager(
            system_prefix=system_prefix,
            workspace_text=workspace_text,
            detail_memory_text=detail_memory_text,
            general_memory=general_memory,
        )

        full_prompt, _ = uncompressed_manager.build_prompt(
            system_prefix=system_prefix,
            workspace_text=workspace_text,
            detail_memory_text=detail_memory_text,
            general_memory=general_memory,
            user_message=prompt,
        )
        compact_prompt, _ = compressed_manager.build_prompt(
            system_prefix=system_prefix,
            workspace_text=workspace_text,
            detail_memory_text=detail_memory_text,
            general_memory=general_memory,
            user_message=prompt,
        )
        original_chars.append(len(full_prompt))
        compressed_chars.append(len(compact_prompt))

    group_count = len(prompts)
    avg_original = sum(original_chars) / group_count
    avg_compressed = sum(compressed_chars) / group_count
    avg_ratio = avg_compressed / avg_original if avg_original else 0.0

    print("测试prompt组数:", group_count)
    print("原prompt平均字数:", f"{avg_original:.2f}")
    print("压缩后平均字数:", f"{avg_compressed:.2f}")
    print("平均压缩率:", f"{avg_ratio:.4f}")


if __name__ == "__main__":
    run()
