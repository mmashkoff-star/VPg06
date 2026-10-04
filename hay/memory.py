"""Долговременный контекст пользователя в PineconeDocumentStore (cosine)."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from haystack import Document
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

logger = logging.getLogger(__name__)


def _openai_secret() -> Secret:
    if os.getenv("OPENAI_API_KEY"):
        return Secret.from_env_var("OPENAI_API_KEY")
    if os.getenv("PROXYAPI_API_KEY"):
        return Secret.from_env_var("PROXYAPI_API_KEY")
    raise ValueError("OPENAI_API_KEY or PROXYAPI_API_KEY is not set")


def _user_filter(user_id: str) -> dict[str, Any]:
    return {"field": "meta.user_id", "operator": "==", "value": user_id}


class PineconeContextMemory:
    """Хранит и достаёт контекст диалога через интеграцию Haystack ↔ Pinecone."""

    def __init__(
        self,
        *,
        index: str | None = None,
        namespace: str | None = None,
        dimension: int | None = None,
        embedding_model: str | None = None,
        top_k: int = 8,
    ) -> None:
        load_dotenv()

        self.index_name = (
            index
            or os.getenv("PINECONE_INDEX_NAME")
            or os.getenv("PINECONE_INDEX")
            or "hay-assistant"
        )
        self.namespace = namespace if namespace is not None else (
            os.getenv("PINECONE_NAMESPACE") or "hay-assistant"
        )
        self.dimension = dimension or int(os.getenv("PINECONE_DIMENSION", "1536"))
        self.embedding_model = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )
        # ProxyAPI иногда требует префикс openai/...
        if self.embedding_model.startswith("openai/"):
            self.embedding_model = self.embedding_model.removeprefix("openai/")

        self.top_k = max(1, top_k)
        api_base_url = os.getenv("OPENAI_BASE_URL") or None
        api_key = _openai_secret()

        cloud = os.getenv("PINECONE_CLOUD", "aws")
        region = os.getenv("PINECONE_REGION", "us-east-1")

        self.document_store = PineconeDocumentStore(
            index=self.index_name,
            namespace=self.namespace,
            metric="cosine",
            dimension=self.dimension,
            spec={"serverless": {"region": region, "cloud": cloud}},
            show_progress=False,
        )

        self._doc_embedder = OpenAIDocumentEmbedder(
            api_key=api_key,
            model=self.embedding_model,
            api_base_url=api_base_url,
            progress_bar=False,
        )
        self._text_embedder = OpenAITextEmbedder(
            api_key=api_key,
            model=self.embedding_model,
            api_base_url=api_base_url,
        )
        self._retriever = PineconeEmbeddingRetriever(document_store=self.document_store)

        logger.info(
            "PineconeContextMemory ready: index=%s namespace=%s dim=%s metric=cosine model=%s",
            self.index_name,
            self.namespace,
            self.dimension,
            self.embedding_model,
        )

    def recall(self, user_id: str, query: str, *, top_k: int | None = None) -> list[str]:
        """Возвращает релевантные фрагменты контекста по косинусному сходству."""
        embedding = self._text_embedder.run(text=query)["embedding"]
        result = self._retriever.run(
            query_embedding=embedding,
            filters=_user_filter(user_id),
            top_k=top_k or self.top_k,
        )
        documents: list[Document] = result.get("documents") or []
        facts: list[str] = []
        seen: set[str] = set()
        for doc in documents:
            content = (doc.content or "").strip()
            if not content or content in seen:
                continue
            seen.add(content)
            facts.append(content)
        logger.info("Memory recall: user_id=%s hits=%d", user_id, len(facts))
        return facts

    def remember(
        self,
        user_id: str,
        text: str,
        *,
        role: str = "context",
        source: str = "chat",
    ) -> str:
        """Индексирует фрагмент контекста пользователя в Pinecone."""
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            raise ValueError("text must be a non-empty string")

        doc_id = str(uuid.uuid4())
        document = Document(
            id=doc_id,
            content=cleaned,
            meta={
                "user_id": user_id,
                "role": role,
                "source": source,
            },
        )
        embedded = self._doc_embedder.run(documents=[document])["documents"]
        self.document_store.write_documents(embedded)
        logger.info(
            "Memory write: user_id=%s role=%s id=%s text=%r",
            user_id,
            role,
            doc_id,
            cleaned if len(cleaned) <= 120 else cleaned[:120] + "...",
        )
        return doc_id

    def remember_user_message(self, user_id: str, user_text: str) -> str:
        """Сохраняет в Pinecone только сообщение пользователя."""
        return self.remember(user_id, user_text, role="user", source="user_message")

    def forget_user(self, user_id: str) -> int:
        """Удаляет весь контекст пользователя."""
        deleted = self.document_store.delete_by_filter(filters=_user_filter(user_id))
        logger.info("Memory forget: user_id=%s deleted=%s", user_id, deleted)
        return int(deleted or 0)
