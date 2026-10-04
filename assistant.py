"""Чат-ассистент: ответ через LLM и долговременная память пользователя в Pinecone."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from pinecone_manager import MemoryWriteResult, PineconeManager


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Ты дружелюбный персональный помощник в Telegram.
Отвечай на языке пользователя, кратко и по делу, если не просят развёрнуто.
Используй блок «Память о пользователе», когда он помогает ответить точнее.
Не выдумывай факты о пользователе: опирайся только на память и текущий диалог.
Если данных не хватает — задай один короткий уточняющий вопрос.
"""

FACT_EXTRACTION_PROMPT = """\
Из сообщения пользователя извлеки информацию, которую стоит сохранить в долговременной памяти.

Запоминай:
- имя, город, работу, предпочтения, цели, ограничения;
- планы, встречи, дедлайны, напоминания и как пользователь просит о них;
- любые устойчивые сведения о самом пользователе.

Правила:
- возвращай только JSON-объект вида {"facts": ["...", "..."]};
- каждый факт — одна короткая фраза от третьего лица;
- не включай только приветствия и общие вопросы без личной информации;
- если запоминать нечего, верни {"facts": []}.
"""


class ChatAssistant:
    """Диалог с пользователем + запись/чтение фактов через PineconeManager."""

    def __init__(
        self,
        memory: PineconeManager | None = None,
        *,
        openai_client: OpenAI | None = None,
        model: str | None = None,
        history_limit: int = 12,
        memory_top_k: int = 8,
    ) -> None:
        load_dotenv()
        self.memory = memory or PineconeManager()
        self.model = model or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
        self.history_limit = max(2, history_limit)
        self.memory_top_k = max(1, memory_top_k)
        self._histories: dict[str, deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=self.history_limit)
        )
        self._openai = openai_client or self._build_openai_client()

    def _build_openai_client(self) -> OpenAI:
        api_key = os.getenv("PROXYAPI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("PROXYAPI_API_KEY or OPENAI_API_KEY is not set")
        kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    @staticmethod
    def user_filter(user_id: str) -> dict[str, Any]:
        return {"user_id": {"$eq": user_id}}

    def clear_history(self, user_id: str) -> None:
        self._histories.pop(user_id, None)

    def recall(self, user_id: str, query: str, *, top_k: int | None = None) -> list[str]:
        """Достаёт релевантные факты о пользователе из векторной памяти."""
        logger.info(
            "Memory recall: user_id=%s top_k=%s query=%r",
            user_id,
            top_k or self.memory_top_k,
            query if len(query) <= 120 else query[:120] + "...",
        )
        response = self.memory.query_by_text(
            query,
            top_k=top_k or self.memory_top_k,
            filter=self.user_filter(user_id),
            include_metadata=True,
        )
        matches = getattr(response, "matches", None) or []
        facts: list[str] = []
        seen: set[str] = set()
        for match in matches:
            metadata = getattr(match, "metadata", None) or {}
            if not isinstance(metadata, Mapping):
                continue
            text = metadata.get("text")
            if not isinstance(text, str):
                continue
            normalized = text.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            facts.append(normalized)
        logger.info("Memory recall result: user_id=%s facts=%d", user_id, len(facts))
        return facts

    def remember_user_fact(
        self,
        user_id: str,
        text: str,
        *,
        source: str = "chat",
    ) -> MemoryWriteResult:
        """Сохраняет факт о пользователе с дедупликацией по косинусному сходству."""
        logger.info(
            "Memory write requested: user_id=%s source=%s text=%r",
            user_id,
            source,
            text if len(text) <= 120 else text[:120] + "...",
        )
        result = self.memory.remember(
            text,
            metadata={"user_id": user_id, "source": source},
            filter=self.user_filter(user_id),
        )
        logger.info(
            "Memory write result: user_id=%s source=%s action=%s id=%s "
            "similarity=%s threshold=%.4f",
            user_id,
            source,
            result.action,
            result.id,
            "none" if result.similarity is None else f"{result.similarity:.4f}",
            result.threshold,
        )
        return result

    def forget_user(self, user_id: str) -> None:
        """Удаляет всю долговременную память пользователя и короткий диалог."""
        logger.info("Memory forget requested: user_id=%s", user_id)
        self.memory.delete(filter=self.user_filter(user_id))
        self.clear_history(user_id)
        logger.info("Memory forget finished: user_id=%s", user_id)

    def remember_profile(
        self,
        user_id: str,
        *,
        full_name: str | None = None,
        username: str | None = None,
    ) -> list[MemoryWriteResult]:
        """Запоминает базовые данные профиля Telegram."""
        results: list[MemoryWriteResult] = []
        if full_name:
            results.append(
                self.remember_user_fact(
                    user_id,
                    f"Имя пользователя в Telegram: {full_name}",
                    source="profile",
                )
            )
        if username:
            results.append(
                self.remember_user_fact(
                    user_id,
                    f"Telegram username: @{username.lstrip('@')}",
                    source="profile",
                )
            )
        return results

    def extract_facts(self, user_message: str) -> list[str]:
        """Просит LLM выделить из сообщения факты, достойные долговременной памяти."""
        logger.info(
            "Fact extraction start: text=%r",
            user_message if len(user_message) <= 120 else user_message[:120] + "...",
        )
        response = self._openai.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": FACT_EXTRACTION_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Fact extraction returned invalid JSON: %r", content)
            return []

        raw_facts = payload.get("facts", []) if isinstance(payload, Mapping) else []
        if not isinstance(raw_facts, Sequence) or isinstance(raw_facts, (str, bytes)):
            logger.warning("Fact extraction payload has no facts list: %r", payload)
            return []

        facts: list[str] = []
        for item in raw_facts:
            if not isinstance(item, str):
                continue
            cleaned = re.sub(r"\s+", " ", item).strip(" -•\t")
            if cleaned:
                facts.append(cleaned)
        logger.info("Fact extraction result: count=%d facts=%s", len(facts), facts)
        return facts

    def _persist_turn(
        self,
        user_id: str,
        user_text: str,
        bot_text: str,
    ) -> list[MemoryWriteResult]:
        """Индексирует реплику пользователя, ответ бота и извлечённые факты."""
        results: list[MemoryWriteResult] = [
            self.remember_user_fact(user_id, user_text, source="user_message"),
            self.remember_user_fact(user_id, bot_text, source="bot_message"),
        ]

        try:
            facts = self.extract_facts(user_text)
        except Exception:
            logger.exception("Fact extraction failed for user_id=%s", user_id)
            return results

        if not facts:
            logger.info(
                "No structured facts extracted for user_id=%s; turn messages already indexed",
                user_id,
            )
            return results

        for fact in facts:
            if fact.casefold() in {user_text.casefold(), bot_text.casefold()}:
                continue
            results.append(self.remember_user_fact(user_id, fact, source="fact"))
        return results

    def reply(self, user_id: str, user_message: str) -> str:
        """Отвечает пользователю, подтягивая и обновляя его память в Pinecone."""
        text = user_message.strip()
        if not text:
            return "Напишите сообщение текстом — так я смогу помочь и запомнить важное."

        memories = self.recall(user_id, text)
        memory_block = (
            "\n".join(f"- {fact}" for fact in memories)
            if memories
            else "Пока нет сохранённых фактов."
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": f"Память о пользователе:\n{memory_block}",
            },
            *self._histories[user_id],
            {"role": "user", "content": text},
        ]

        completion = self._openai.chat.completions.create(
            model=self.model,
            temperature=0.4,
            messages=messages,
        )
        answer = (completion.choices[0].message.content or "").strip()
        if not answer:
            answer = "Я здесь. Уточните, пожалуйста, чем помочь."

        history = self._histories[user_id]
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": answer})

        try:
            self._persist_turn(user_id, text, answer)
        except Exception:
            logger.exception("Failed to index conversation turn for user_id=%s", user_id)

        return answer
