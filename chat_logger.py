"""Логирование работы бота и запись переписки в файлы."""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent / "logs"
BOT_LOG_FILE = LOG_DIR / "bot.log"
CONVERSATIONS_LOG_FILE = LOG_DIR / "conversations.log"
CHATS_DIR = LOG_DIR / "chats"


def setup_logging(*, level: int = logging.INFO) -> None:
    """Настраивает логирование в консоль и в ``logs/bot.log``."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Не дублируем хендлеры при повторном вызове (например, при reload).
    has_file = any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")).resolve() == BOT_LOG_FILE.resolve()
        for handler in root.handlers
    )
    if not has_file:
        file_handler = logging.FileHandler(BOT_LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    has_stream = any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in root.handlers)
    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    # Меньше шума от сторонних библиотек в файле и консоли.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("telebot").setLevel(logging.INFO)


class ConversationLogger:
    """Пишет переписку в общий файл и в файл каждого пользователя."""

    def __init__(
        self,
        *,
        conversations_file: Path = CONVERSATIONS_LOG_FILE,
        chats_dir: Path = CHATS_DIR,
    ) -> None:
        self.conversations_file = conversations_file
        self.chats_dir = chats_dir
        self._lock = threading.Lock()
        self.conversations_file.parent.mkdir(parents=True, exist_ok=True)
        self.chats_dir.mkdir(parents=True, exist_ok=True)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _format_line(
        self,
        *,
        role: str,
        user_id: str,
        text: str,
        username: str | None = None,
        full_name: str | None = None,
    ) -> str:
        who = full_name or ""
        if username:
            who = f"{who} @{username.lstrip('@')}".strip()
        identity = f"user_id={user_id}"
        if who:
            identity = f"{identity} ({who})"
        safe_text = text.replace("\r\n", "\n").replace("\r", "\n")
        return f"[{self._timestamp()}] {identity} {role}: {safe_text}\n"

    def log(
        self,
        *,
        role: str,
        user_id: str,
        text: str,
        username: str | None = None,
        full_name: str | None = None,
    ) -> None:
        line = self._format_line(
            role=role,
            user_id=user_id,
            text=text,
            username=username,
            full_name=full_name,
        )
        user_file = self.chats_dir / f"{user_id}.log"
        with self._lock:
            with self.conversations_file.open("a", encoding="utf-8") as shared:
                shared.write(line)
            with user_file.open("a", encoding="utf-8") as personal:
                personal.write(line)

    def log_user(
        self,
        *,
        user_id: str,
        text: str,
        username: str | None = None,
        full_name: str | None = None,
    ) -> None:
        self.log(
            role="USER",
            user_id=user_id,
            text=text,
            username=username,
            full_name=full_name,
        )

    def log_bot(
        self,
        *,
        user_id: str,
        text: str,
        username: str | None = None,
        full_name: str | None = None,
    ) -> None:
        self.log(
            role="BOT",
            user_id=user_id,
            text=text,
            username=username,
            full_name=full_name,
        )
