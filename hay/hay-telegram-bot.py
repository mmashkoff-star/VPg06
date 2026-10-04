"""Telegram-бот: Haystack Agent + PineconeDocumentStore + PyTelegramBotAPI."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import telebot
from dotenv import load_dotenv
from telebot import types

# Позволяет запускать файл напрямую: python hay/hay-telegram-bot.py
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant import AssistantReply, HaystackAssistant  # noqa: E402
from chat_logger import ConversationLogger, setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger("hay_telegram_bot")
chat_log = ConversationLogger()

BOT_COMMANDS = [
    types.BotCommand("start", "Начать работу с помощником"),
    types.BotCommand("help", "Справка по возможностям"),
    types.BotCommand("memory", "Что бот помнит о вас"),
    types.BotCommand("forget", "Очистить долговременный контекст"),
    types.BotCommand("reset", "Сбросить короткий диалог"),
    types.BotCommand("dogfact", "Случайный факт о собаках"),
    types.BotCommand("dogpic", "Случайная собака + описание породы"),
    types.BotCommand("weather", "Погода и фото города"),
]


def setup_bot_menu(bot: telebot.TeleBot) -> None:
    bot.set_my_commands(BOT_COMMANDS)
    bot.set_chat_menu_button(menu_button=types.MenuButtonCommands(type="commands"))
    logger.info(
        "Command menu: %s",
        ", ".join(f"/{item.command}" for item in BOT_COMMANDS),
    )


def build_bot() -> tuple[telebot.TeleBot, HaystackAssistant]:
    load_dotenv(ROOT.parent / ".env")
    load_dotenv(ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY") and os.getenv("PROXYAPI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["PROXYAPI_API_KEY"]

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set")

    bot = telebot.TeleBot(token, parse_mode=None)
    assistant = HaystackAssistant()
    return bot, assistant


def _user_id(message: types.Message) -> str:
    user = message.from_user
    if user is not None and getattr(user, "id", None) is not None:
        return str(user.id)
    return str(message.chat.id)


def _clean(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def _display_name(user: types.User | None) -> str | None:
    if user is None:
        return None
    parts = [part for part in (_clean(user.first_name), _clean(user.last_name)) if part]
    return " ".join(parts) or None


def _username(user: types.User | None) -> str | None:
    if user is None or not user.username:
        return None
    return user.username.strip().lstrip("@") or None


def _greeting_name(user: types.User | None) -> str:
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
    """Отправляет ответ. Служебные ответы команд не пишутся в переписку."""
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
    text: str | None = None,
    save: bool = True,
) -> dict[str, str | None]:
    meta = _user_meta(message)
    content = text if text is not None else (message.text or f"<{message.content_type}>")
    if save:
        chat_log.log_user(text=content, **meta)
    logger.info(
        "Incoming %s from user_id=%s username=%s full_name=%s: %s",
        kind,
        meta["user_id"],
        meta["username"] or "-",
        meta["full_name"] or "-",
        content if len(content) <= 200 else content[:200] + "...",
    )
    return meta


def _send_assistant_reply(
    bot: telebot.TeleBot,
    message: types.Message,
    reply: AssistantReply,
) -> None:
    _reply(bot, message, reply.text, service=False)
    if reply.image_url:
        caption = (reply.image_caption or "Фото").strip()
        if len(caption) > 1000:
            caption = caption[:997] + "..."
        try:
            bot.send_photo(message.chat.id, reply.image_url, caption=caption)
            meta = _user_meta(message)
            chat_log.log_bot(
                text=f"[photo] {reply.image_url}",
                **meta,
            )
            logger.info(
                "Photo sent to user_id=%s: %s",
                meta["user_id"],
                reply.image_url,
            )
        except Exception:
            logger.exception("Failed to send photo: %s", reply.image_url)
            _reply(
                bot,
                message,
                f"Не удалось вложить фото, вот ссылка: {reply.image_url}",
                service=False,
            )


def register_handlers(bot: telebot.TeleBot, assistant: HaystackAssistant) -> None:
    @bot.message_handler(commands=["start"])
    def handle_start(message: types.Message) -> None:
        meta = _log_incoming(message, kind="command", save=False)
        user_id = meta["user_id"] or ""
        full_name = meta["full_name"]
        username = meta["username"]
        try:
            assistant.remember_profile(user_id, full_name=full_name, username=username)
        except Exception:
            logger.exception("Failed to store profile for user_id=%s", user_id)

        _reply(
            bot,
            message,
            (
                f"Привет, {_greeting_name(message.from_user)}! Я персональный помощник "
                "на Haystack Agent с памятью в Pinecone.\n\n"
                "Пишите как обычно — я учитываю ваш контекст и продолжаю диалог.\n\n"
                "Ещё умею:\n"
                "• случайный факт о собаках\n"
                "• случайную картинку собаки + описание породы через OpenAI Vision\n"
                "• погоду в городе и фото города\n\n"
                "Команды: /help /memory /forget /reset /dogfact /dogpic /weather"
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
                "Я умный персональный помощник.\n\n"
                "• обычные сообщения — диалог с учётом контекста из Pinecone "
                "(cosine similarity)\n"
                "• /dogfact — факт о собаках (внешний API)\n"
                "• /dogpic — картинка собаки + vision-описание породы\n"
                "• /weather Казань — погода и фото города\n"
                "• /memory — что я о вас помню\n"
                "• /forget — очистить долговременный контекст\n"
                "• /reset — сбросить короткий диалог"
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
                "профиль пользователя факты предпочтения имя город работа цели диалог",
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
        _reply(bot, message, f"Вот что я о вас помню:\n\n{lines}", service=True)

    @bot.message_handler(commands=["forget"])
    def handle_forget(message: types.Message) -> None:
        meta = _log_incoming(message, kind="command", save=False)
        user_id = meta["user_id"] or ""
        try:
            assistant.forget_user(user_id)
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
            "Контекст очищен. Можем начать знакомство заново.",
            service=True,
        )

    @bot.message_handler(commands=["reset"])
    def handle_reset(message: types.Message) -> None:
        meta = _log_incoming(message, kind="command", save=False)
        assistant.clear_history(meta["user_id"] or "")
        _reply(
            bot,
            message,
            "Короткий диалог сброшен. Долговременный контекст в Pinecone сохранён.",
            service=True,
        )

    @bot.message_handler(commands=["dogfact"])
    def handle_dogfact(message: types.Message) -> None:
        _process_user_text(
            bot,
            assistant,
            message,
            "Расскажи случайный факт о собаках",
            kind="command",
        )

    @bot.message_handler(commands=["dogpic"])
    def handle_dogpic(message: types.Message) -> None:
        _process_user_text(
            bot,
            assistant,
            message,
            "Покажи случайную картинку собаки и опиши породу с краткой предысторией",
            kind="command",
        )

    @bot.message_handler(commands=["weather"])
    def handle_weather(message: types.Message) -> None:
        parts = (message.text or "").split(maxsplit=1)
        city = parts[1].strip() if len(parts) > 1 else ""
        if city:
            prompt = (
                f"Покажи текущую погоду в городе {city} и фото этого города. "
                "Используй инструмент get_weather_and_city_photo."
            )
        else:
            prompt = (
                "Пользователь хочет узнать погоду. Город не указан — "
                "спроси, в каком городе показать погоду и фото."
            )
        _process_user_text(bot, assistant, message, prompt, kind="command")

    @bot.message_handler(func=lambda message: True, content_types=["text"])
    def handle_text(message: types.Message) -> None:
        text = (message.text or "").strip()
        if not text:
            return
        _process_user_text(bot, assistant, message, text, kind="text")

    @bot.message_handler(content_types=["photo", "document", "voice", "sticker", "video"])
    def handle_unsupported(message: types.Message) -> None:
        _log_incoming(message, kind=message.content_type, save=False)
        _reply(
            bot,
            message,
            "Пока я понимаю только текстовые сообщения. Напишите, пожалуйста, словами.",
            service=True,
        )


def _process_user_text(
    bot: telebot.TeleBot,
    assistant: HaystackAssistant,
    message: types.Message,
    text: str,
    *,
    kind: str = "text",
) -> None:
    meta = _log_incoming(message, kind=kind, text=text, save=True)
    user_id = meta["user_id"] or ""
    bot.send_chat_action(message.chat.id, "typing")
    try:
        reply = assistant.reply(user_id, text)
    except Exception:
        logger.exception("Failed to reply for user_id=%s", user_id)
        _reply(
            bot,
            message,
            "Сейчас не получилось ответить. Попробуйте ещё раз через минуту.",
            service=True,
        )
        return

    _send_assistant_reply(bot, message, reply)


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

    logger.info(
        "Haystack Telegram bot started. Polling... "
        "Logs: hay/logs/bot.log, chats: hay/logs/conversations.log"
    )
    bot.infinity_polling(skip_pending=True, timeout=60, long_polling_timeout=60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
