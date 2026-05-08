from __future__ import annotations

import shutil
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, TextIO


@dataclass(frozen=True)
class UIColors:
    reset: str = "\x1b[0m"
    dim: str = "\x1b[2m"
    bold: str = "\x1b[1m"
    slate: str = "\x1b[38;5;244m"
    ocean: str = "\x1b[38;5;117m"
    mint: str = "\x1b[38;5;43m"
    amber: str = "\x1b[38;5;214m"
    red: str = "\x1b[38;5;203m"


class TerminalUI:
    """Terminal interaction UI with Claude-like structure and thinking animation."""

    _frames = (".", "..", "...", "....")
    _spinner = "|/-\\"

    def __init__(self, stream: TextIO | None = None, interval: float = 0.1):
        self.stream = stream or sys.stdout
        self.interval = interval
        self.colors = UIColors()
        self._pause_event = threading.Event()
        self._render_lock = threading.Lock()

    def _is_tty(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def _supports_color(self) -> bool:
        return self._is_tty()

    def _w(self) -> int:
        try:
            return max(72, shutil.get_terminal_size(fallback=(100, 24)).columns)
        except Exception:
            return 100

    def _paint(self, text: str, color: str) -> str:
        if not self._supports_color():
            return text
        return f"{color}{text}{self.colors.reset}"

    def _rule(self, char: str = "-") -> str:
        return char * min(self._w(), 100)

    def _write(self, text: str = "") -> None:
        with self._render_lock:
            self.stream.write(text + "\n")
            self.stream.flush()

    @contextmanager
    def pause_effects(self) -> Iterator[None]:
        """Temporarily pause terminal animations for interactive input."""
        if not self._is_tty():
            yield
            return

        self._pause_event.set()
        with self._render_lock:
            self.stream.write("\r" + " " * 120 + "\r")
            self.stream.flush()
        try:
            yield
        finally:
            self._pause_event.clear()

    def print_welcome(
        self, workspace: str, session_id: str, approval_mode: str
    ) -> None:
        title = self._paint("CODARA TERMINAL", self.colors.bold + self.colors.ocean)
        subtitle = self._paint("Coding Agent Interface", self.colors.slate)

        self._write(self._paint(self._rule("="), self.colors.ocean))
        self._write(f"{title}  {subtitle}")
        self._write(self._paint(self._rule("-"), self.colors.slate))
        self._write(
            "  "
            + self._paint("workspace", self.colors.slate)
            + ": "
            + self._paint(workspace, self.colors.mint)
        )
        self._write(
            "  "
            + self._paint("session", self.colors.slate)
            + ": "
            + self._paint(session_id, self.colors.mint)
        )
        self._write(
            "  "
            + self._paint("approval", self.colors.slate)
            + ": "
            + self._paint(approval_mode, self.colors.amber)
        )
        self._write(self._paint(self._rule("="), self.colors.ocean))
        self._write(
            self._paint(
                "输入 /help 查看命令；直接输入需求开始工作。", self.colors.slate
            )
        )

    def print_help(self, help_text: str) -> None:
        self._write(self._paint(self._rule("-"), self.colors.slate))
        self._write(self._paint(help_text, self.colors.slate))
        self._write(self._paint(self._rule("-"), self.colors.slate))

    def print_user(self, text: str) -> None:
        tag = self._paint("YOU", self.colors.bold + self.colors.ocean)
        self._write(f"\n[{tag}] {text}")

    def print_assistant(self, text: str) -> None:
        tag = self._paint("CODARA", self.colors.bold + self.colors.mint)
        self._write(f"\n[{tag}]\n{text}")

    def print_system(self, text: str) -> None:
        tag = self._paint("SYSTEM", self.colors.bold + self.colors.amber)
        self._write(f"\n[{tag}] {text}")

    def print_error(self, text: str) -> None:
        tag = self._paint("ERROR", self.colors.bold + self.colors.red)
        self._write(f"\n[{tag}] {text}")

    def prompt(self) -> str:
        with self.pause_effects():
            if self._supports_color():
                prefix = self._paint("codara", self.colors.bold + self.colors.ocean)
                marker = self._paint(">", self.colors.bold + self.colors.mint)
                return input(f"\n{prefix} {marker} ").strip()
            return input("\ncodara> ").strip()

    def confirm(self, prompt: str) -> bool:
        with self.pause_effects():
            raw = (
                input(self._paint(prompt + " (y/n): ", self.colors.amber))
                .strip()
                .lower()
            )
        return raw in {"y", "yes"}

    @contextmanager
    def thinking(self, message: str = "思考中") -> Iterator[None]:
        if not self._is_tty():
            self.stream.write(f"[thinking] {message}\n")
            self.stream.flush()
            try:
                yield
            finally:
                self.stream.write("[thinking] 完成\n")
                self.stream.flush()
            return

        stop_event = threading.Event()

        def _spin() -> None:
            i = 0
            while not stop_event.is_set():
                if self._pause_event.is_set():
                    stop_event.wait(self.interval)
                    continue
                dots = self._frames[i % len(self._frames)]
                spin = self._spinner[i % len(self._spinner)]
                line = f"{spin} {message}{dots}"
                with self._render_lock:
                    painted = self._paint(line, self.colors.slate)
                    self.stream.write("\r" + painted + " " * 8)
                    self.stream.flush()
                i += 1
                stop_event.wait(self.interval)

        thread = threading.Thread(target=_spin, daemon=True)
        thread.start()

        failed = False
        try:
            yield
        except Exception:
            failed = True
            raise
        finally:
            stop_event.set()
            thread.join(timeout=0.3)
            status = "失败" if failed else "完成"
            tone = self.colors.red if failed else self.colors.mint
            with self._render_lock:
                self.stream.write("\r" + " " * 120 + "\r")
                done = self._paint(f"{message} ({status})", tone)
                self.stream.write(done + "\n")
                self.stream.flush()
