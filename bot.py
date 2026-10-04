"""Telegram-бот-помощник на pyTelegramBotAPI с памятью в Pinecone."""

from __future__ import annotations

import logging
import os
import sys

import telebot
from dotenv import load_dotenv
from telebot import types

from assistant import ChatAssistant
from chat_logger import ConversationLogger, setup_logging


setup_logging()
logger = logging.getLogger("telegram_bot")
chat_log = ConversationLogger()

BOT_COMMANDS = [
    types.BotCommand("start", "Начать работу с ботом"),
    types.BotCommand("help", "Справка по возможностям"),
    types.BotCommand("memory", "Что бот о вас помнит"),
    types.BotCommand("forget", "Очистить долговременную память"),
    types.BotCommand("reset", "Сбросить короткий диалог"),
]


def setup_bot_menu(bot: telebot.TeleBot) -> None:
    """Регистрирует меню команд в интерфейсе Telegram."""
    bot.set_my_commands(BOT_COMMANDS)
    bot.set_chat_menu_button(menu_button=types.MenuButtonCommands(type="commands"))
    logger.info(
        "Bot command menu registered: %s",
        ", ".join(f"/{command.command}" for command in BOT_COMMANDS),
    )


def build_bot() -> tuple[telebot.TeleBot, ChatAssistant]:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set")

    bot = telebot.TeleBot(token, parse_mode=None)
    assistant = ChatAssistant()
    return bot, assistant


def _user_id(message: types.Message) -> str:
    user = message.from_user
    if user is not None and getattr(user, "id", None) is not None:
        return str(user.id)
    return str(message.chat.id)


def _clean_person_name(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def _display_name(user: types.User | None) -> str | None:
    """Собирает имя из first_name/last_name; None, если обоих нет."""
    if user is None:
        return None
    parts = [
        part
        for part in (
            _clean_person_name(user.first_name),
            _clean_person_name(user.last_name),
        )
        if part
    ]
    return " ".join(parts) or None


def _username(user: types.User | None) -> str | None:
    if user is None or not user.username:
        return None
    cleaned = user.username.strip().lstrip("@")
    return cleaned or None


def _greeting_name(user: types.User | None) -> str:
    """Имя для приветствия: ФИО → @username → нейтральное обращение."""
    full_name = _display_name(user)
    if full_name:
        return full_name
    username = _username(user)
    if username:
        return f"@{username}"
    return "пользователь"


def _user_meta(message: types.Message) -> dict[str, str | None]:
    user = message.from_user
    return {
        "user_id": _user_id(message),
        "username": _username(user),
        "full_name": _display_name(user),
    }


def _reply(
    bot: telebot.TeleBot,
    message: types.Message,
    text: str,
    *,
    service: bool = False,
) -> None:
    """Отправляет ответ. Служебные ответы не пишутся в сохранённую переписку."""
    meta = _user_meta(message)
    bot.reply_to(message, text)
    preview = text if len(text) <= 200 else text[:200] + "..."
    if service:
        logger.info(
            "Service reply (not saved) to user_id=%s: %s",
            meta["user_id"],
            preview,
        )
        return

    chat_log.log_bot(text=text, **meta)
    logger.info("Reply to user_id=%s: %s", meta["user_id"], preview)


def _log_incoming(
    message: types.Message,
    *,
    kind: str = "message",
    save: bool = True,
) -> dict[str, str | None]:
    meta = _user_meta(message)
    text = message.text or f"<{message.content_type}>"
    if save:
        chat_log.log_user(text=text, **meta)
    logger.info(
        "Incoming %s from user_id=%s username=%s full_name=%s: %s",
        kind,
        meta["user_id"],
        meta["username"] or "-",
        meta["full_name"] or "-",
        text if len(text) <= 200 else text[:200] + "...",
    )
    return meta


def register_handlers(bot: telebot.TeleBot, assistant: ChatAssistant) -> None:
    @bot.message_handler(commands=["start"])
    def handle_start(message: types.Message) -> None:
        meta = _log_incoming(message, kind="command", save=False)
        user_id = meta["user_id"] or ""
        full_name = meta["full_name"]
        username = meta["username"]

        if full_name or username:
            try:
                assistant.remember_profile(
                    user_id,
                    full_name=full_name,
                    username=username,
                )
                logger.info(
                    "Profile stored for user_id=%s full_name=%s username=%s",
                    user_id,
                    full_name or "-",
                    username or "-",
                )
            except Exception:
                logger.exception("Failed to store profile for user_id=%s", user_id)
        else:
            logger.info(
                "Profile skipped for user_id=%s: no first/last name and no username",
                user_id,
            )

        _reply(
            bot,
            message,
            (
                f"Привет, {_greeting_name(message.from_user)}! Я персональный помощник.\n\n"
                "Пишите мне как обычно — я отвечу и запомню важное о вас "
                "в долговременной памяти.\n\n"
                "Команды:\n"
                "/help — справка\n"
                "/memory — что я о вас помню\n"
                "/forget — очистить память\n"
                "/reset — сбросить короткий диалог"
            ),
            service=True,
        )

    @bot.message_handler(commands=["help"])
    def handle_help(message: types.Message) -> None:
        _log_incoming(message, kind="command", save=False)
        _reply(
            bot,
            message,
            (
                "Я чат-помощник с долговременной памятью.\n\n"
                "Просто пишите сообщения — я отвечаю и сохраняю устойчивые факты "
                "о вас (имя, город, предпочтения и т.п.).\n\n"
                "/memory — показать сохранённые факты\n"
                "/forget — удалить всю память о вас\n"
                "/reset — начать диалог заново без очистки памяти"
            ),
            service=True,
        )

    @bot.message_handler(commands=["memory"])
    def handle_memory(message: types.Message) -> None:
        meta = _log_incoming(message, kind="command", save=False)
        user_id = meta["user_id"] or ""
        bot.send_chat_action(message.chat.id, "typing")
        try:
            facts = assistant.recall(
                user_id,
                "профиль пользователя факты предпочтения имя город работа цели",
                top_k=20,
            )
        except Exception:
            logger.exception("Failed to recall memory for user_id=%s", user_id)
            _reply(
                bot,
                message,
                "Не удалось прочитать память. Попробуйте позже.",
                service=True,
            )
            return

        if not facts:
            _reply(bot, message, "Пока я ничего о вас не запомнил.", service=True)
            return

        lines = "\n".join(f"• {fact}" for fact in facts)
        logger.info("Recalled %d facts for user_id=%s", len(facts), user_id)
        _reply(bot, message, f"Вот что я о вас помню:\n\n{lines}", service=True)

    @bot.message_handler(commands=["forget"])
    def handle_forget(message: types.Message) -> None:
        meta = _log_incoming(message, kind="command", save=False)
        user_id = meta["user_id"] or ""
        try:
            assistant.forget_user(user_id)
            logger.info("Memory cleared for user_id=%s", user_id)
        except Exception:
            logger.exception("Failed to forget memory for user_id=%s", user_id)
            _reply(
                bot,
                message,
                "Не удалось очистить память. Попробуйте позже.",
                service=True,
            )
            return
        _reply(
            bot,
            message,
            "Память о вас очищена. Можем начать знакомство заново.",
            service=True,
        )

    @bot.message_handler(commands=["reset"])
    def handle_reset(message: types.Message) -> None:
        meta = _log_incoming(message, kind="command", save=False)
        user_id = meta["user_id"] or ""
        assistant.clear_history(user_id)
        logger.info("Short history reset for user_id=%s", user_id)
        _reply(
            bot,
            message,
            "Короткий диалог сброшен. Долговременная память сохранена.",
            service=True,
        )

    @bot.message_handler(func=lambda message: True, content_types=["text"])
    def handle_text(message: types.Message) -> None:
        text = (message.text or "").strip()
        if not text:
            return

        meta = _log_incoming(message, kind="text")
        user_id = meta["user_id"] or ""
        bot.send_chat_action(message.chat.id, "typing")
        try:
            answer = assistant.reply(user_id, text)
        except Exception:
            logger.exception("Failed to reply for user_id=%s", user_id)
            _reply(
                bot,
                message,
                "Сейчас не получилось ответить. Попробуйте ещё раз через минуту.",
                service=True,
            )
            return

        # Обычный ответ диалога сохраняется в переписку; в Pinecone он уже
        # проиндексирован внутри assistant.reply.
        _reply(bot, message, answer, service=False)

    @bot.message_handler(content_types=["photo", "document", "voice", "sticker", "video"])
    def handle_unsupported(message: types.Message) -> None:
        _log_incoming(message, kind=message.content_type, save=False)
        _reply(
            bot,
            message,
            "Пока я понимаю только текстовые сообщения. Напишите, пожалуйста, словами.",
            service=True,
        )


def main() -> int:
    try:
        bot, assistant = build_bot()
    except Exception as exc:
        logger.error("Bot startup failed: %s", exc)
        return 1

    register_handlers(bot, assistant)
    try:
        setup_bot_menu(bot)
    except Exception:
        logger.exception("Failed to register Telegram command menu")

    logger.info("Bot started. Polling... Logs: logs/bot.log, chats: logs/conversations.log")
    bot.infinity_polling(skip_pending=True, timeout=60, long_polling_timeout=60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
