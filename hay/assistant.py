"""Haystack tool-calling Agent — персональный помощник с памятью в Pinecone."""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.chat_message import ChatRole
from haystack.utils import Secret

from memory import PineconeContextMemory
from tools import ToolSession, create_assistant_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Ты умный персональный помощник в Telegram.

Твоя роль:
- вести естественный диалог как внимательный помощник;
- обязательно учитывать контекст пользователя из блока «Память / контекст»;
- помнить предпочтения, факты и прошлые темы разговора;
- отвечать дружелюбно и по делу.

Язык ответа (строго):
- отвечай целиком на том же языке, на котором написал пользователь;
- если пользователь пишет по-русски — весь ответ только на чистом русском;
- не смешивай языки в одном предложении;
- не вставляй английские корни в русские слова (запрещены формы вроде «fascinирующая», «amazingовый»);
- используй нормальные слова выбранного языка: «увлекательная», «интересная», «поразительная» и т.п.;
- иностранные имена, названия API/брендов и устойчивые термины можно оставлять как есть, если нет привычного перевода.

Инструменты:
- get_random_dog_fact — случайный факт о собаках из внешнего API;
- get_random_dog_image_and_describe — случайная картинка собаки + vision-описание породы;
- get_weather_and_city_photo — текущая погода в городе и фото города (нужен город).

Правила:
- вызывай tools только когда они реально нужны запросу пользователя;
- для погоды обязательно вызывай get_weather_and_city_photo и передавай город;
- если пользователь просит погоду, но город не указан — спроси город;
- не выдумывай факты о пользователе: опирайся на память и текущий диалог;
- если данных мало — задай один короткий уточняющий вопрос;
- когда используешь картинку собаки, включи в ответ описание породы и её предысторию;
- для погоды кратко перескажи температуру, ощущения, влажность, ветер и состояние неба;
- факты из внешних API пересказывай на языке пользователя, не копируй сырой английский без перевода.
"""


@dataclass(slots=True)
class AssistantReply:
    text: str
    image_url: str | None = None
    image_caption: str | None = None


def _openai_secret() -> Secret:
    if os.getenv("OPENAI_API_KEY"):
        return Secret.from_env_var("OPENAI_API_KEY")
    if os.getenv("PROXYAPI_API_KEY"):
        return Secret.from_env_var("PROXYAPI_API_KEY")
    raise ValueError("OPENAI_API_KEY or PROXYAPI_API_KEY is not set")


def _chat_model() -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return model.removeprefix("openai/")


def _ensure_openai_env() -> None:
    """Haystack читает OPENAI_API_KEY; подставляем ProxyAPI при необходимости."""
    if not os.getenv("OPENAI_API_KEY") and os.getenv("PROXYAPI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["PROXYAPI_API_KEY"]


class HaystackAssistant:
    """Персональный помощник: Agent + Pinecone-контекст + Telegram tools."""

    def __init__(
        self,
        memory: PineconeContextMemory | None = None,
        *,
        history_limit: int = 10,
        memory_top_k: int = 8,
    ) -> None:
        load_dotenv()
        _ensure_openai_env()

        self.memory = memory or PineconeContextMemory(top_k=memory_top_k)
        self.history_limit = max(2, history_limit)
        self._histories: dict[str, deque[ChatMessage]] = defaultdict(
            lambda: deque(maxlen=self.history_limit)
        )
        self.session = ToolSession()
        tools = create_assistant_tools(self.session)

        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(
                api_key=_openai_secret(),
                model=_chat_model(),
                api_base_url=os.getenv("OPENAI_BASE_URL") or None,
                generation_kwargs={"temperature": 0.4},
            ),
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            exit_conditions=["text"],
            max_agent_steps=8,
            raise_on_tool_invocation_failure=False,
        )
        self.agent.warm_up()
        logger.info("HaystackAssistant ready with %d tools", len(tools))

    def clear_history(self, user_id: str) -> None:
        self._histories.pop(user_id, None)

    def forget_user(self, user_id: str) -> int:
        self.clear_history(user_id)
        return self.memory.forget_user(user_id)

    def recall(self, user_id: str, query: str, *, top_k: int | None = None) -> list[str]:
        return self.memory.recall(user_id, query, top_k=top_k)

    def remember_profile(
        self,
        user_id: str,
        *,
        full_name: str | None = None,
        username: str | None = None,
    ) -> None:
        if full_name:
            self.memory.remember(
                user_id,
                f"Имя пользователя в Telegram: {full_name}",
                role="profile",
                source="profile",
            )
        if username:
            self.memory.remember(
                user_id,
                f"Telegram username: @{username.lstrip('@')}",
                role="profile",
                source="profile",
            )

    def reply(self, user_id: str, user_message: str) -> AssistantReply:
        text = user_message.strip()
        if not text:
            return AssistantReply(
                text="Напишите сообщение текстом — так я смогу помочь и учесть контекст."
            )

        self.session.reset()
        memories = self.memory.recall(user_id, text)
        memory_block = (
            "\n".join(f"- {item}" for item in memories)
            if memories
            else "Пока нет сохранённого контекста."
        )

        messages: list[ChatMessage] = [
            ChatMessage.from_system(
                "Память / контекст пользователя (из Pinecone, cosine similarity):\n"
                f"{memory_block}"
            ),
            *self._histories[user_id],
            ChatMessage.from_user(text),
        ]

        try:
            output = self.agent.run(messages=messages)
        except Exception:
            logger.exception("Agent run failed for user_id=%s", user_id)
            raise

        answer = self._extract_answer(output)
        image_url = self.session.last_image_url
        image_caption = self.session.last_image_caption

        history = self._histories[user_id]
        history.append(ChatMessage.from_user(text))
        history.append(ChatMessage.from_assistant(answer))

        try:
            self.memory.remember_user_message(user_id, text)
        except Exception:
            logger.exception("Failed to persist user message for user_id=%s", user_id)

        return AssistantReply(
            text=answer,
            image_url=image_url,
            image_caption=image_caption,
        )

    @staticmethod
    def _extract_answer(output: dict[str, Any]) -> str:
        if not isinstance(output, dict):
            return "Я здесь. Уточните, пожалуйста, чем помочь."

        last_message = output.get("last_message")
        if isinstance(last_message, ChatMessage) and last_message.text:
            return last_message.text.strip()

        messages = output.get("messages") or []
        for message in reversed(messages):
            if (
                isinstance(message, ChatMessage)
                and message.text
                and message.is_from(ChatRole.ASSISTANT)
            ):
                return message.text.strip()

        return "Я здесь. Уточните, пожалуйста, чем помочь."
