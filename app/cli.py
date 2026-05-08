from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .context_manager import ContextManager
from .models import DeepSeekChatClient, ModelError
from .rewrite import RequestRewriter
from .run_store import RunStore
from .runtime import CodaraRuntime
from .tools import ToolRegistry
from .ui import TerminalUI


WELCOME = """
Codara MVP 已启动
输入普通文本即可对话
内部命令：
/help      查看帮助
/approval  查看或切换审批模式（ask/auto）
/memory    查看结构化记忆摘要
/session   查看当前会话 ID
/reset     清空本次会话状态
/exit      退出
""".strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _build_approval_callback(mode: str, ui: TerminalUI | None = None):
    mode = mode.strip().lower()
    if mode == "auto":

        def _approve_auto(_tool_name: str, _args: dict) -> bool:
            return True

        return _approve_auto

    def _approve_ask(tool_name: str, args: dict) -> bool:
        """
        高风险工具需要人类审批执行
        """
        if ui is not None:
            with ui.pause_effects():
                ui.print_system(f"即将执行高风险工具: {tool_name}")
                ui.print_system(f"参数: {args}")
                return ui.confirm("是否允许执行")

        print(f"\n[审批] 即将执行高风险工具: {tool_name}")
        print(f"[参数] {args}")
        ans = input("是否允许执行？(y/n): ").strip().lower()
        return ans in {"y", "yes"}

    return _approve_ask


def _set_approval_mode(
    runtime: CodaraRuntime, mode: str, ui: TerminalUI | None = None
) -> str:
    normalized = mode.strip().lower()
    if normalized not in {"ask", "auto"}:
        return "审批模式无效，仅支持 ask 或 auto。"

    runtime.tool_registry.approval_callback = _build_approval_callback(normalized, ui)
    return f"审批模式已切换为: {normalized}"


def build_runtime_from_env(ui: TerminalUI | None = None) -> CodaraRuntime:
    # 从项目根目录 .env 加载环境变量
    load_dotenv()

    workspace_root = Path(os.getenv("CODARA_WORKSPACE", os.getcwd())).resolve()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    max_steps = int(os.getenv("CODARA_MAX_STEPS", "16"))
    max_retries = int(os.getenv("CODARA_MAX_RETRIES", "6"))
    approval_mode = os.getenv("CODARA_APPROVAL", "ask").strip().lower()

    model = DeepSeekChatClient(
        api_key=api_key,
        model="deepseek-chat",
        base_url=base_url,
        timeout=60,
        temperature=0.2,
    )

    tools = ToolRegistry(
        workspace_root=workspace_root,
        approval_callback=_build_approval_callback(approval_mode, ui),
    )
    store = RunStore(workspace_root=workspace_root)
    ctx_mgr = ContextManager()
    rewriter = RequestRewriter(model_client=model)

    runtime = CodaraRuntime.restore_or_new(
        workspace_root=workspace_root,
        model_client=model,
        tool_registry=tools,
        run_store=store,
        context_manager=ctx_mgr,
        request_rewriter=rewriter,
    )
    runtime.max_steps = max_steps
    runtime.max_retries = max_retries
    return runtime


def main() -> int:
    ui = TerminalUI()

    try:
        runtime = build_runtime_from_env(ui)
    except ModelError as e:
        ui.print_error(f"启动失败: {e}")
        ui.print_system("请先设置 DEEPSEEK_API_KEY。")
        return 1
    except Exception as e:
        ui.print_error(f"启动失败: {e}")
        return 1

    approval_mode = os.getenv("CODARA_APPROVAL", "ask").strip().lower()
    if approval_mode not in {"ask", "auto"}:
        approval_mode = "ask"
    ui.print_welcome(str(runtime.workspace_root), runtime.session_id, approval_mode)

    while True:
        try:
            user_input = ui.prompt()
        except (EOFError, KeyboardInterrupt):
            ui.print_system("退出。")
            return 0

        if not user_input:
            continue

        if user_input == "/exit":
            ui.print_system("退出。")
            return 0
        if user_input == "/help":
            ui.print_help(WELCOME)
            continue
        if user_input == "/approval":
            ui.print_system(f"当前审批模式: {approval_mode}")
            continue
        if user_input.startswith("/approval "):
            _, _, mode = user_input.partition(" ")
            message = _set_approval_mode(runtime, mode, ui)
            if message.startswith("审批模式已切换为:"):
                approval_mode = mode.strip().lower()
            ui.print_system(message)
            continue
        if user_input == "/session":
            ui.print_system(f"current session_id: {runtime.session_id}")
            continue
        if user_input == "/reset":
            runtime.reset()
            ui.print_system(f"会话已重置，新 session_id: {runtime.session_id}")
            continue
        if user_input == "/memory":
            ui.print_system(runtime.memory.render_for_prompt(user_input))
            continue

        try:
            ui.print_user(user_input)

            with ui.thinking("优化请求中"):
                rewritten_input = runtime.rewrite_user_message(user_input)

            if rewritten_input:
                if rewritten_input == user_input.strip():
                    ui.print_system("重写后的请求（无变化）：")
                else:
                    ui.print_system("重写后的请求：")
                ui.print_system(rewritten_input)

            with ui.thinking("Codara 正在思考"):
                answer = runtime.ask(rewritten_input, already_rewritten=True)
            ui.print_assistant(answer)
        except Exception as e:
            ui.print_error(f"运行错误: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
