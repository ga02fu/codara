from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_manager import ContextManager
from .memory import LayeredMemory
from .models import DeepSeekChatClient, ModelError
from .rewrite import RequestRewriter
from .run_store import RunStore
from .task_state import TaskState
from .tools import ToolError, ToolRegistry
from .workspace import WorkspaceContext


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ParsedAction:
    kind: str
    final_text: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] | None = None
    retry_reason: str = ""


class CodaraRuntime:
    """
    运行时主循环。
    用单轮 ask 驱动完整链路：感知-决策-工具-记录-停止
    """

    def __init__(
        self,
        workspace_root: Path,
        model_client: DeepSeekChatClient,
        tool_registry: ToolRegistry,
        memory: LayeredMemory,
        run_store: RunStore,
        context_manager: ContextManager,
        request_rewriter: RequestRewriter | None = None,
        session_id: str | None = None,
        max_steps: int = 16,
        max_retries: int = 2,
    ):
        self.workspace_root = workspace_root.resolve()
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.memory = memory
        self.run_store = run_store
        self.context_manager = context_manager
        self.request_rewriter = request_rewriter
        self.max_steps = max_steps
        self.max_retries = max_retries

        self.session_id = session_id or self._new_session_id()
        self.general_memory: list[dict[str, str]] = []
        self.trace: list[dict[str, Any]] = []
        self.workspace_ctx = WorkspaceContext.build(
            self.workspace_root
        )  # 工作空间上下文

        # 把工具记忆总结接入同一 deepseek-chat 模型。
        self.memory.summarize_tool_fn = self._llm_summarize_tool_note

    def _new_session_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def ask(self, user_message: str, already_rewritten: bool = False) -> str:
        """
        ask 入口，建立本轮状态并执行主循环
        """
        prepared_message = (
            str(user_message).strip()
            if already_rewritten
            else self._prepare_user_message(user_message)
        )

        # 新一轮用户请求，先清理工具去重状态，避免跨轮误判重复调用。
        if hasattr(self.tool_registry, "reset_round_state"):
            self.tool_registry.reset_round_state()

        self.memory.set_task_summary(prepared_message)
        self.general_memory.append({"role": "user", "content": prepared_message})

        task = TaskState(
            run_id=uuid.uuid4().hex,
            task_id=uuid.uuid4().hex[:8],
        )

        try:
            answer = self._run_loop(task, prepared_message)
            self.general_memory.append({"role": "assistant", "content": answer})
            self._save_session(task)
            return answer
        except Exception as e:
            task.fail(str(e))
            self._save_session(task)
            raise

    def rewrite_user_message(self, user_message: str) -> str:
        """
        对外暴露请求重写能力，供 CLI 在执行前展示重写结果。
        """
        return self._prepare_user_message(user_message)

    def _prepare_user_message(self, user_message: str) -> str:
        text = str(user_message).strip()
        if not text:
            return text
        if self.request_rewriter is None:
            return text

        try:
            rewritten = self.request_rewriter.rewrite(text).strip()
            if rewritten and rewritten != text:
                self.trace.append(
                    {
                        "type": "request_rewrite",
                        "at": now_iso(),
                        "data": {
                            "original": text[:1200],
                            "rewritten": rewritten[:1800],
                        },
                    }
                )
            return rewritten or text
        except Exception as e:
            # 重写失败不阻塞主流程，直接使用原请求继续执行。
            self.trace.append(
                {
                    "type": "request_rewrite_error",
                    "at": now_iso(),
                    "data": str(e)[:800],
                }
            )
            return text

    def _run_loop(self, task: TaskState, user_message: str) -> str:
        """
        主循环入口，codara采用单轮ask驱动简单agent loop，没有复杂的编排
        """
        recoverable_errors: list[str] = []
        runtime_hints: list[str] = []

        def _add_runtime_hint(text: str) -> None:
            msg = str(text).strip()
            if not msg:
                return
            if msg not in runtime_hints:
                runtime_hints.append(msg)

        while True:
            if task.tool_steps >= self.max_steps:
                task.stop("step_limit_reached")
                return "已达到本轮工具调用上限，请你给我更聚焦的下一步指令。"

            if task.attempts > self.max_retries:
                task.stop("retry_limit_reached")
                return (
                    "本轮任务未收敛，已达到尝试上限。请缩小目标或给出更明确的下一步。"
                )

            prompt, prompt_meta = self.context_manager.build_prompt(
                system_prefix=self._system_prefix(user_message, runtime_hints),
                workspace_text=self.workspace_ctx.render(),
                detail_memory_text=self.memory.render_for_prompt(user_message),
                general_memory=self.general_memory,
                user_message=user_message,
            )
            self.trace.append(
                {"type": "prompt_meta", "at": now_iso(), "data": prompt_meta}
            )

            try:
                completion = self.model_client.complete(
                    prompt=prompt, max_new_tokens=1200
                )
            except ModelError as e:
                task.stop("model_error")
                return f"模型调用失败: {e}"

            raw = completion.text.strip()
            self.trace.append(
                {"type": "model_raw", "at": now_iso(), "data": raw[:2000]}
            )

            action = self._parse_action(raw)

            if action.kind == "final":
                task.finish("final_answer_returned")
                return action.final_text.strip()

            if action.kind == "retry":
                task.attempts += 1
                reason = action.retry_reason.strip() or "模型请求继续思考一轮。"
                self.trace.append(
                    {
                        "type": "model_retry",
                        "at": now_iso(),
                        "data": reason[:800],
                    }
                )
                _add_runtime_hint(
                    f"提示：收到 retry 请求（原因：{reason}）。请在下一轮优先收敛到 tool 或 final。"
                )
                continue

            if action.kind == "tool":
                task.tool_steps += 1
                tool_name = action.tool_name
                tool_args = action.tool_args or {}

                try:
                    result = self.tool_registry.execute(tool_name, tool_args)
                    self.memory.update_after_tool(tool_name, tool_args, result)
                    # 工具结果写入 general_memory，使用摘要避免冗余代码片段。
                    self._append_general_memory(
                        "tool",
                        self._build_tool_memory_entry(tool_name, tool_args, result),
                    )
                    if (
                        tool_name == "list_files"
                        and str(tool_args.get("path", ".")).strip() == "."
                    ):
                        _add_runtime_hint(
                            "提示：根目录已检索完成；下一步应进入子目录或读取目标文件，不要重复 list_files(path='.')。",
                        )
                    if tool_name == "list_files" and self._is_write_like_request(
                        user_message
                    ):
                        _add_runtime_hint(
                            (
                                "提示：当前任务是创建/写入类请求，目录确认后必须尽快转为 write_file。"
                                "禁止连续重复 list_files。"
                            ),
                        )
                    self.trace.append(
                        {
                            "type": "tool_result",
                            "at": now_iso(),
                            "data": {
                                "name": tool_name,
                                "args": tool_args,
                                "result": result[:2000],
                            },
                        }
                    )

                    # 对创建/写入类任务做快速收敛，避免完成后仍反复探测触发上限。
                    if self._is_write_like_request(user_message) and tool_name in {
                        "write_file",
                        "patch_file",
                    }:
                        if recoverable_errors:
                            _add_runtime_hint(
                                (
                                    "提示：可执行写入已完成，但本轮存在前序失败项。"
                                    "请立即输出 <final>，同时说明已完成内容与失败内容，不要只返回写入成功。"
                                ),
                            )
                            continue

                        task.finish("write_task_completed")
                        return (
                            "已完成请求：目标文件已写入并保存。\n"
                            f"工具结果：{result[:300]}"
                        )

                    continue
                except ToolError as e:
                    # 仅在工具失败时计入一次重试。
                    task.attempts += 1
                    err_text = f"工具执行失败: {e}"

                    if any(
                        token in str(e)
                        for token in (
                            "文件不存在",
                            "目录不存在",
                            "不是目录",
                            "未匹配到",
                        )
                    ):
                        recoverable_errors.append(err_text)
                        _add_runtime_hint(
                            (
                                "提示：当前错误可恢复。请继续完成其他可执行子任务，"
                                "并在最终 <final> 中明确列出已完成项、失败项及失败原因。"
                            ),
                        )

                    if "连续重复工具调用" in str(e):
                        if recoverable_errors:
                            self._append_general_memory("tool", err_text)
                            _add_runtime_hint(
                                (
                                    "提示：不要继续重复同一探测工具。"
                                    "请立刻输出 <final>，给出部分完成结果（完成项 + 失败项）。"
                                ),
                            )
                            continue

                        if (
                            tool_name == "list_files"
                            and str(tool_args.get("path", ".")).strip() == "."
                        ):
                            fallback_args = {"path": "app"}
                            try:
                                fallback_result = self.tool_registry.execute(
                                    "list_files", fallback_args
                                )
                                self.memory.update_after_tool(
                                    "list_files", fallback_args, fallback_result
                                )
                                _add_runtime_hint(
                                    "提示：检测到重复调用 list_files(path='.')，已自动切换为 list_files(path='app')。",
                                )
                                self._append_general_memory(
                                    "tool",
                                    self._build_tool_memory_entry(
                                        "list_files", fallback_args, fallback_result
                                    ),
                                )
                                self.trace.append(
                                    {
                                        "type": "tool_result",
                                        "at": now_iso(),
                                        "data": {
                                            "name": "list_files",
                                            "args": fallback_args,
                                            "result": fallback_result[:2000],
                                        },
                                    }
                                )
                                continue
                            except ToolError:
                                pass

                        self._append_general_memory("tool", err_text)
                        task.stop("duplicate_tool_loop")
                        return (
                            "检测到模型重复调用同一工具并被拦截，已提前停止本轮。"
                            "请给出更具体路径/关键词，或直接指定下一步工具参数。"
                        )

                    self._append_general_memory("tool", err_text)

                    self.trace.append(
                        {"type": "tool_error", "at": now_iso(), "data": err_text}
                    )
                    continue

            # 若解析失败，给模型一次格式纠正机会
            task.attempts += 1
            _add_runtime_hint(
                (
                    '提示：输出不合规。请只返回一种标签：<tool>{"name":"...","args":{}}</tool>'
                    " 或 <final>...</final>。若部分子任务失败但其余已完成，请像正常完成任务一样给出高质量结论。"
                ),
            )

    def _append_general_memory(self, role: str, content: str) -> None:
        # system 类信息用于当轮引导，不进入 general_memory，避免污染长期对话轨迹。
        if str(role).strip().lower() == "system":
            return
        text = str(content).strip()
        if not text:
            return
        item = {"role": role, "content": text}
        if self.general_memory and self.general_memory[-1] == item:
            return
        self.general_memory.append(item)

    def _build_tool_memory_entry(
        self, tool_name: str, tool_args: dict[str, Any], result: str
    ) -> str:
        args_text = json.dumps(tool_args, ensure_ascii=False)
        tool_note = (
            self.memory.relevant_notes[-1]
            if self.memory.relevant_notes
            else f"{tool_name} 已执行"
        )

        preview = ""
        if tool_name in {"list_files", "read_file", "search"}:
            compact = str(result).replace("\n", " | ").strip()
            if compact:
                preview = compact[:220]

        if preview:
            return f"{tool_name}({args_text}): {tool_note} [preview] {preview}"
        return f"{tool_name}({args_text}): {tool_note}"

    def _is_write_like_request(self, user_message: str) -> bool:
        text = user_message.strip().lower()
        keywords = ["新建", "创建", "写", "写入", "保存", "文件", "md", "markdown"]
        return any(k in text for k in keywords)

    def _llm_summarize_tool_note(
        self, tool_name: str, args: dict[str, Any], result: str
    ) -> str:
        """
        用 deepseek-chat 将工具结果压缩为短摘要。
        要求模型严格输出 JSON，避免格式漂移污染记忆。
        """
        args_text = json.dumps(args, ensure_ascii=False)
        result_text = str(result)[:1200]
        prompt = (
            "你是记忆摘要器。只输出严格 JSON，不要解释。\n"
            '输出格式: {"summary":"..."}\n'
            "约束:\n"
            "1) summary 必须是单行文本\n"
            "2) summary 最多 160 个字符\n"
            "3) 保留关键事实: 工具名、关键路径/参数、结果状态\n"
            "4) 不要虚构\n"
            "5) 禁止输出代码、行号清单、长片段原文\n"
            "6) 风格必须是自然语言摘要，不是原文拷贝\n"
            "7) 不得输出 XML/HTML 标签，不得输出 <tool>/<final>\n"
            "7) 优先使用如下风格:\n"
            "   - list_files: 相关文件包含 ...\n"
            "   - read_file: 文件xxx主要内容：...\n"
            "   - write_file: 已写入xxx，主要变更：...\n"
            f"tool_name={tool_name}\n"
            f"args={args_text}\n"
            f"result={result_text}\n"
        )
        try:
            completion = self.model_client.complete(prompt=prompt, max_new_tokens=180)
            raw = completion.text.strip()
            data = json.loads(raw)
            summary = str(data.get("summary", "")).replace("\n", " ").strip()
            if summary:
                lowered = summary.lower()
                if (
                    "from __future__" in lowered
                    or "import " in lowered
                    or "class " in lowered
                    or "def " in lowered
                    or "[project]" in lowered
                ):
                    return f"{tool_name} 已执行，已生成自然语言摘要。"
                return summary[:160]
        except Exception:
            pass

        # 兜底仅保留通用短句，不使用规则化模板。
        return f"{tool_name} 已执行，摘要生成失败。"

    def _system_prefix(
        self, user_message: str, runtime_hints: list[str] | None = None
    ) -> str:
        tools = self.tool_registry.list_tools()
        tool_lines = [f"- {t['name']} (risky={t['risky']})" for t in tools]
        write_task_rule = ""
        hints_block = ""
        if runtime_hints:
            lines = [f"- {h}" for h in runtime_hints[-6:]]
            hints_block = "\n【本轮运行提示】\n" + "\n".join(lines) + "\n"
        if self._is_write_like_request(user_message):
            write_task_rule = (
                "\n【写入任务专用约束】\n"
                "- 目录存在性最多验证 1 次；验证后下一步必须 write_file 或 patch_file。\n"
                "- 若用户要求在 docs 下新建文件，优先 path='app/docs/<filename>'。\n"
                "- 禁止对同一路径连续重复 list_files/read_file。\n"
            )
        return (
            "你是 Codara。你必须严格按链路执行，不得跳过或循环。\n"
            "当前输入已由请求重写器整理为“用户请求 + 子任务”结构。\n"
            "你必须优先按子任务序号推进，逐项执行并记录结果。\n"
            "【输出协议】\n"
            "- 每轮只允许输出一种标签：<tool>...</tool>、<retry>...</retry> 或 <final>...</final>。\n"
            '- 需要调用工具时，唯一合法格式：<tool>{"name":"工具名","args":{...}}</tool>\n'
            "- 当本轮无需调用工具但仍需继续推理时，输出 <retry>继续原因</retry>。\n"
            "- 能直接回答时，输出 <final>你的答案</final>。\n"
            "- 当请求含多个子任务时，允许部分完成；必须在 <final> 中说明已完成项与失败项。\n"
            "- 生成 <final> 时，按子任务编号输出每项状态（完成/失败）与简要依据。\n"
            "\n【执行链路】\n"
            "1) 先解析“用户请求/子任务”并锁定当前要执行的子任务编号。\n"
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
            f"{hints_block}"
            f"{write_task_rule}"
            "\n可用工具：\n" + "\n".join(tool_lines)
        )

    def _parse_action(self, raw: str) -> ParsedAction:
        final_match = re.search(r"<final>(.*?)</final>", raw, flags=re.S | re.I)
        if final_match:
            return ParsedAction(kind="final", final_text=final_match.group(1).strip())

        retry_match = re.search(r"<retry>(.*?)(?:</retry>|$)", raw, flags=re.S | re.I)
        if retry_match:
            return ParsedAction(kind="retry", retry_reason=retry_match.group(1).strip())

        tool_match = re.search(r"<tool>(.*?)(?:</tool>|$)", raw, flags=re.S | re.I)
        if tool_match:
            body = tool_match.group(1).strip()
            try:
                payload = json.loads(body)
                name = str(payload.get("name", "")).strip()
                args = payload.get("args", {})
                if not isinstance(args, dict):
                    raise ValueError("args 必须是对象")
                if not name:
                    raise ValueError("name 不能为空")
                return ParsedAction(kind="tool", tool_name=name, tool_args=args)
            except Exception:
                # 兼容模型偶发输出的 XML 子标签格式。
                name_match = re.search(r"<name>(.*?)</name>", body, flags=re.S | re.I)
                path_match = re.search(r"<path>(.*?)</path>", body, flags=re.S | re.I)
                if name_match:
                    name = name_match.group(1).strip()
                    args: dict[str, Any] = {}
                    if path_match:
                        args["path"] = path_match.group(1).strip()
                    return ParsedAction(kind="tool", tool_name=name, tool_args=args)
                return ParsedAction(kind="unknown")

        # MVP 容错：若模型没包标签，直接视作最终答案
        return ParsedAction(kind="final", final_text=raw)

    def _save_session(self, task: TaskState) -> None:
        payload = {
            "version": 1,
            "session_id": self.session_id,
            "workspace_root": str(self.workspace_root),
            "saved_at": now_iso(),
            "task_state": task.to_dict(),
            "general_memory": self.general_memory[-60:],
            "detail_memory": self.memory.to_dict(),
            "trace": self.trace[-200:],
        }
        self.run_store.save(self.session_id, payload)

    def reset(self) -> None:
        self.general_memory.clear()
        self.trace.clear()
        self.memory = LayeredMemory()
        self.memory.summarize_tool_fn = self._llm_summarize_tool_note
        self.session_id = self._new_session_id()

    @classmethod
    def restore_or_new(
        cls,
        workspace_root: Path,
        model_client: DeepSeekChatClient,
        tool_registry: ToolRegistry,
        run_store: RunStore,
        context_manager: ContextManager,
        request_rewriter: RequestRewriter | None = None,
    ) -> "CodaraRuntime":
        latest = run_store.latest_session_id()
        if not latest:
            return cls(
                workspace_root=workspace_root,
                model_client=model_client,
                tool_registry=tool_registry,
                memory=LayeredMemory(),
                run_store=run_store,
                context_manager=context_manager,
                request_rewriter=request_rewriter,
            )

        data = run_store.load(latest)
        if not data:
            return cls(
                workspace_root=workspace_root,
                model_client=model_client,
                tool_registry=tool_registry,
                memory=LayeredMemory(),
                run_store=run_store,
                context_manager=context_manager,
                request_rewriter=request_rewriter,
            )

        runtime = cls(
            workspace_root=workspace_root,
            model_client=model_client,
            tool_registry=tool_registry,
            memory=LayeredMemory.from_dict(
                data.get("detail_memory", data.get("memory", {}))
            ),
            run_store=run_store,
            context_manager=context_manager,
            request_rewriter=request_rewriter,
            session_id=data.get("session_id"),
        )
        runtime.general_memory = data.get("general_memory", data.get("history", []))
        runtime.trace = data.get("trace", [])
        return runtime
