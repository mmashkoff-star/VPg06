"""Менеджер чтения и записи в векторную базу Pinecone."""

from __future__ import annotations

import logging
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


logger = logging.getLogger(__name__)

# Граница косинусного сходства между новым сообщением и уже сохранённым фрагментом.
# Ниже порога сходство низкое: это новая информация, её записываем в новый слот.
# На пороге и выше сходство высокое: совпавший текст — дубликат, запись пропускается;
# другой текст — вариация, обновляется существующий слот.
# Косинус лежит в диапазоне от -1 до 1. Выше порог — чаще создаются новые слоты.
COSINE_SIMILARITY_THRESHOLD = 0.85

MemoryAction = Literal["stored", "updated", "skipped"]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Косинусное сходство двух векторов. 1 — одинаковое направление, 0 — ортогональны."""
    if len(left) != len(right):
        raise ValueError(
            "Cannot compute cosine similarity: "
            f"vector lengths differ ({len(left)} and {len(right)})"
        )
    if not left:
        raise ValueError("Cannot compute cosine similarity of empty vectors")

    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        norm_left += a * a
        norm_right += b * b
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_left) * math.sqrt(norm_right))


def memory_action(
    similarity: float | None,
    *,
    same_text: bool,
    threshold: float,
) -> MemoryAction:
    """Решает, что делать с сообщением относительно уже сохранённых фрагментов."""
    if similarity is None or similarity < threshold:
        return "stored"
    if same_text:
        return "skipped"
    return "updated"


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    """Итог записи сообщения в долговременную память."""

    action: MemoryAction
    id: str
    similarity: float | None
    threshold: float


VectorLike = (
    tuple[str, Sequence[float]]
    | tuple[str, Sequence[float], Mapping[str, Any]]
    | Mapping[str, Any]
)


class PineconeManager:
    """Управляет записью и чтением данных в индексе Pinecone.

    Поддерживает три семейства API Pinecone SDK 10:
    - vectors — ``upsert`` / ``query`` (векторы считаете сами);
    - records — ``upsert_records`` / ``search`` (эмбеддинг на стороне Pinecone);
    - documents — ``documents.*`` (schema-based индексы).

    Долговременная память чат-бота пишется через :meth:`remember`: перед записью
    сообщение сравнивается с уже сохранёнными фрагментами по косинусному сходству.
    Порог задаёт ``COSINE_SIMILARITY_THRESHOLD`` в начале модуля.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        load_dotenv()

        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is not set")

        self.index_name = (
            index_name
            or os.getenv("PINECONE_INDEX_NAME")
            or os.getenv("PINECONE_INDEX")
        )
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME (or PINECONE_INDEX) is not set")

        env_namespace = os.getenv("PINECONE_NAMESPACE")
        self.namespace = namespace if namespace is not None else (env_namespace or "")

        self.embedding_model = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "openai/text-embedding-3-small",
        )
        self._openai_api_key = (
            openai_api_key
            or os.getenv("PROXYAPI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self._openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")

        self._pc = Pinecone(api_key=self.api_key)
        self._index = self._pc.index(name=self.index_name)
        self._openai: OpenAI | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_namespace(self, namespace: str | None, *, require_non_empty: bool = False) -> str:
        resolved = self.namespace if namespace is None else namespace
        if require_non_empty and (not resolved or not str(resolved).strip()):
            raise ValueError(
                "namespace must be a non-empty string for this Pinecone API; "
                "pass namespace=... or set PINECONE_NAMESPACE"
            )
        return resolved

    def _get_openai(self) -> OpenAI:
        if self._openai is None:
            if not self._openai_api_key:
                raise ValueError(
                    "PROXYAPI_API_KEY or OPENAI_API_KEY is required for local embeddings"
                )
            kwargs: dict[str, Any] = {"api_key": self._openai_api_key}
            if self._openai_base_url:
                kwargs["base_url"] = self._openai_base_url
            self._openai = OpenAI(**kwargs)
        return self._openai

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Считает эмбеддинги через OpenAI/ProxyAPI."""
        if not texts:
            return []
        client = self._get_openai()
        response = client.embeddings.create(model=self.embedding_model, input=list(texts))
        return [item.embedding for item in response.data]

    def embed_text(self, text: str) -> list[float]:
        """Считает эмбеддинг одного текста."""
        return self.embed_texts([text])[0]

    @staticmethod
    def _normalize_memory_text(text: str) -> str:
        return " ".join(text.split()).casefold()

    @staticmethod
    def _match_attr(match: Any, name: str) -> Any:
        if isinstance(match, Mapping):
            return match.get(name)
        return getattr(match, name, None)

    def _resolve_similarity_threshold(self, threshold: float | None) -> float:
        value = COSINE_SIMILARITY_THRESHOLD if threshold is None else threshold
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("COSINE_SIMILARITY_THRESHOLD must be a number from -1 to 1")
        resolved = float(value)
        if not -1.0 <= resolved <= 1.0:
            raise ValueError("COSINE_SIMILARITY_THRESHOLD must be between -1 and 1")
        return resolved

    def _closest_memory(
        self,
        vector: Sequence[float],
        *,
        namespace: str | None,
        filter: Mapping[str, Any] | None,
        candidate_k: int,
    ) -> tuple[str | None, float | None, dict[str, Any]]:
        """Ищет сохранённый фрагмент с наибольшим косинусным сходством."""
        if candidate_k < 1:
            raise ValueError("candidate_k must be >= 1")

        response = self.query_by_vector(
            vector,
            top_k=candidate_k,
            namespace=namespace,
            filter=filter,
            include_values=True,
            include_metadata=True,
        )
        matches = self._match_attr(response, "matches") or []

        best_id: str | None = None
        best_similarity: float | None = None
        best_metadata: dict[str, Any] = {}

        for match in matches:
            match_id = self._match_attr(match, "id")
            if not isinstance(match_id, str) or not match_id:
                continue

            values = self._match_attr(match, "values") or []
            metadata = self._match_attr(match, "metadata")
            if not values:
                fetched = self.fetch_vectors([match_id], namespace=namespace)
                stored = self._match_attr(fetched, "vectors") or {}
                stored_vector = stored.get(match_id) if isinstance(stored, Mapping) else None
                if stored_vector is not None:
                    values = self._match_attr(stored_vector, "values") or []
                    if not metadata:
                        metadata = self._match_attr(stored_vector, "metadata")
            if not values:
                continue

            similarity = cosine_similarity(vector, values)
            if best_similarity is None or similarity > best_similarity:
                best_id = match_id
                best_similarity = similarity
                best_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}

        return best_id, best_similarity, best_metadata

    def remember(
        self,
        text: str,
        *,
        id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        filter: Mapping[str, Any] | None = None,
        text_field: str = "text",
        threshold: float | None = None,
        candidate_k: int = 5,
    ) -> MemoryWriteResult:
        """Пишет сообщение в долговременную память с проверкой косинусного сходства.

        Порог — глобальная ``COSINE_SIMILARITY_THRESHOLD`` (или ``threshold`` на один вызов).
        Сходство считается между эмбеддингом сообщения и ближайшими сохранёнными фрагментами.

        * ниже порога — новая информация, создаётся новый слот;
        * на пороге и выше, текст тот же — дубликат, запись пропускается;
        * на пороге и выше, текст другой — вариация, обновляется найденный слот.

        ``filter`` ограничивает сравнение, например памятью одного пользователя:
        ``{"user_id": {"$eq": "42"}}``. Тот же ``user_id`` передайте в ``metadata``,
        чтобы следующие сообщения сравнивались внутри этой памяти.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if id is not None and (not isinstance(id, str) or not id):
            raise ValueError("id must be a non-empty string when provided")
        if not isinstance(text_field, str) or not text_field:
            raise ValueError("text_field must be a non-empty string")

        limit = self._resolve_similarity_threshold(threshold)
        preview = text if len(text) <= 120 else text[:120] + "..."
        logger.info(
            "Pinecone memory check start: text=%r threshold=%.4f namespace=%r filter=%s",
            preview,
            limit,
            self._resolve_namespace(namespace),
            filter,
        )

        embedding = self.embed_text(text)
        matched_id, similarity, matched_metadata = self._closest_memory(
            embedding,
            namespace=namespace,
            filter=filter,
            candidate_k=candidate_k,
        )

        previous_text = matched_metadata.get(text_field)
        same_text = (
            isinstance(previous_text, str)
            and self._normalize_memory_text(previous_text) == self._normalize_memory_text(text)
        )
        action = memory_action(similarity, same_text=same_text, threshold=limit)
        if matched_id is None:
            action = "stored"

        similarity_repr = "none" if similarity is None else f"{similarity:.4f}"
        compared = (
            "no stored match"
            if matched_id is None
            else f"id={matched_id!r} same_text={same_text}"
        )
        if similarity is None:
            threshold_cmp = "no similarity (empty memory or no candidates)"
        elif similarity < limit:
            threshold_cmp = f"{similarity_repr} < {limit:.4f} → low similarity → new slot"
        else:
            threshold_cmp = (
                f"{similarity_repr} >= {limit:.4f} → high similarity → "
                + ("duplicate skip" if same_text else "update existing slot")
            )

        logger.info(
            "Pinecone threshold decision: similarity=%s threshold=%.4f compared=%s decision=%s",
            similarity_repr,
            limit,
            compared,
            action,
        )

        if action == "skipped":
            logger.info(
                "Pinecone write skipped (duplicate): id=%s similarity=%s threshold=%.4f text=%r",
                matched_id,
                similarity_repr,
                limit,
                preview,
            )
            return MemoryWriteResult(
                action="skipped",
                id=matched_id,
                similarity=similarity,
                threshold=limit,
            )

        slot_id = matched_id if action == "updated" and matched_id is not None else (id or str(uuid.uuid4()))
        stored_metadata: dict[str, Any] = {}
        if action == "updated":
            stored_metadata.update(matched_metadata)
        if metadata:
            stored_metadata.update(dict(metadata))
        stored_metadata[text_field] = text
        stored_metadata = {
            key: value for key, value in stored_metadata.items() if value is not None
        }

        logger.info(
            "Pinecone write %s: id=%s similarity=%s threshold=%.4f metadata_keys=%s text=%r",
            action,
            slot_id,
            similarity_repr,
            limit,
            sorted(stored_metadata),
            preview,
        )
        self.upsert_vectors(
            [{"id": slot_id, "values": embedding, "metadata": stored_metadata}],
            namespace=namespace,
            show_progress=False,
        )
        logger.info("Pinecone upsert finished: action=%s id=%s", action, slot_id)
        return MemoryWriteResult(
            action=action,
            id=slot_id,
            similarity=similarity,
            threshold=limit,
        )

    # ------------------------------------------------------------------
    # Write: vectors
    # ------------------------------------------------------------------

    def upsert_vectors(
        self,
        vectors: Sequence[VectorLike],
        *,
        namespace: str | None = None,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> Any:
        """Записывает готовые векторы в индекс.

        Каждый элемент: ``Vector``, ``(id, values)``, ``(id, values, metadata)``
        или ``{"id", "values", "metadata"?}``.
        """
        return self._index.upsert(
            vectors=list(vectors),
            namespace=self._resolve_namespace(namespace),
            batch_size=batch_size,
            show_progress=show_progress,
        )

    def upsert_texts_as_vectors(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        text_field: str = "text",
        id_field: str = "id",
        namespace: str | None = None,
        batch_size: int | None = 100,
        metadata_fields: Sequence[str] | None = None,
    ) -> Any:
        """Эмбеддит тексты локально и пишет их как векторы.

        Ожидает элементы вида ``{"id": "...", "text": "...", ...metadata}``.
        """
        if not items:
            raise ValueError("items must be a non-empty sequence")

        texts: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for i, item in enumerate(items):
            item_id = item.get(id_field) or item.get("_id")
            text = item.get(text_field)
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(f"Item at index {i} must contain a non-empty '{id_field}'")
            if not isinstance(text, str) or not text:
                raise ValueError(f"Item at index {i} must contain a non-empty '{text_field}'")

            ids.append(item_id)
            texts.append(text)

            if metadata_fields is None:
                metadata = {
                    key: value
                    for key, value in item.items()
                    if key not in {id_field, "_id", text_field} and value is not None
                }
            else:
                metadata = {
                    key: item[key]
                    for key in metadata_fields
                    if key in item and item[key] is not None
                }
            # Сохраняем исходный текст в метаданных для удобного чтения.
            metadata.setdefault(text_field, text)
            metadatas.append(metadata)

        embeddings = self.embed_texts(texts)
        vectors = [
            {"id": item_id, "values": values, "metadata": metadata}
            for item_id, values, metadata in zip(ids, embeddings, metadatas, strict=True)
        ]
        return self.upsert_vectors(vectors, namespace=namespace, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Write: records (integrated inference)
    # ------------------------------------------------------------------

    def upsert_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        namespace: str | None = None,
    ) -> Any:
        """Записывает records; эмбеддинг считает Pinecone на сервере.

        Каждый record должен содержать ``_id``/``id`` и текстовое поле
        из field_map индекса (часто ``text`` или ``chunk_text``).
        """
        return self._index.upsert_records(
            records=list(records),
            namespace=self._resolve_namespace(namespace, require_non_empty=True),
        )

    # ------------------------------------------------------------------
    # Write: documents (schema-based)
    # ------------------------------------------------------------------

    def upsert_documents(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        namespace: str | None = None,
    ) -> Any:
        """Записывает documents в schema-based индекс."""
        return self._index.documents.upsert(
            namespace=self._resolve_namespace(namespace, require_non_empty=True),
            documents=list(documents),
        )

    def batch_upsert_documents(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        namespace: str | None = None,
        batch_size: int = 50,
        show_progress: bool = True,
    ) -> Any:
        """Пакетная запись documents."""
        return self._index.documents.batch_upsert(
            namespace=self._resolve_namespace(namespace, require_non_empty=True),
            documents=list(documents),
            batch_size=batch_size,
            show_progress=show_progress,
        )

    # ------------------------------------------------------------------
    # Read: by vector
    # ------------------------------------------------------------------

    def query_by_vector(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 5,
        namespace: str | None = None,
        filter: Mapping[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> Any:
        """Поиск ближайших соседей по готовому вектору."""
        return self._index.query(
            vector=list(vector),
            top_k=top_k,
            namespace=self._resolve_namespace(namespace),
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
        )

    def query_by_id(
        self,
        id: str,
        *,
        top_k: int = 5,
        namespace: str | None = None,
        filter: Mapping[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> Any:
        """Поиск ближайших соседей по ID уже сохранённого вектора."""
        return self._index.query(
            id=id,
            top_k=top_k,
            namespace=self._resolve_namespace(namespace),
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
        )

    def search_records_by_vector(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 5,
        namespace: str | None = None,
        filter: Mapping[str, Any] | None = None,
        fields: Sequence[str] | None = None,
        rerank: Mapping[str, Any] | None = None,
    ) -> Any:
        """Поиск records по вектору (integrated inference API)."""
        return self._index.search(
            namespace=self._resolve_namespace(namespace, require_non_empty=True),
            top_k=top_k,
            vector=list(vector),
            filter=filter,
            fields=fields,
            rerank=rerank,
        )

    def search_documents_by_vector(
        self,
        vector: Sequence[float],
        *,
        field: str,
        top_k: int = 5,
        namespace: str | None = None,
        filter: Mapping[str, Any] | None = None,
        include_fields: Sequence[str] | None = None,
    ) -> Any:
        """Поиск documents по dense-вектору.

        ``field`` — имя dense_vector-поля в схеме индекса.
        """
        return self._index.documents.search(
            namespace=self._resolve_namespace(namespace, require_non_empty=True),
            top_k=top_k,
            score_by=[{"type": "dense_vector", "field": field, "values": list(vector)}],
            filter=filter,
            include_fields=include_fields,
        )

    # ------------------------------------------------------------------
    # Read: by text
    # ------------------------------------------------------------------

    def query_by_text(
        self,
        text: str,
        *,
        top_k: int = 5,
        namespace: str | None = None,
        filter: Mapping[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> Any:
        """Эмбеддит текст локально и ищет через vector ``query``."""
        vector = self.embed_text(text)
        return self.query_by_vector(
            vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
        )

    def search_by_text(
        self,
        text: str,
        *,
        top_k: int = 5,
        namespace: str | None = None,
        filter: Mapping[str, Any] | None = None,
        fields: Sequence[str] | None = None,
        rerank: Mapping[str, Any] | None = None,
        input_field: str = "text",
    ) -> Any:
        """Поиск records по тексту; эмбеддинг на стороне Pinecone."""
        return self._index.search(
            namespace=self._resolve_namespace(namespace, require_non_empty=True),
            top_k=top_k,
            inputs={input_field: text},
            filter=filter,
            fields=fields,
            rerank=rerank,
        )

    def search_documents_by_text(
        self,
        text: str,
        *,
        fields: Sequence[str],
        top_k: int = 5,
        namespace: str | None = None,
        filter: Mapping[str, Any] | None = None,
        include_fields: Sequence[str] | None = None,
    ) -> Any:
        """Поиск documents по тексту (BM25/text scoring)."""
        return self._index.documents.search(
            namespace=self._resolve_namespace(namespace, require_non_empty=True),
            top_k=top_k,
            score_by=[{"type": "text", "query": text, "fields": list(fields)}],
            filter=filter,
            include_fields=include_fields,
        )

    # ------------------------------------------------------------------
    # Fetch / update / delete / stats
    # ------------------------------------------------------------------

    def fetch_vectors(
        self,
        ids: Sequence[str],
        *,
        namespace: str | None = None,
    ) -> Any:
        """Читает векторы по списку ID."""
        return self._index.fetch(ids=list(ids), namespace=self._resolve_namespace(namespace))

    def fetch_documents(
        self,
        ids: Sequence[str],
        *,
        namespace: str | None = None,
        include_fields: Sequence[str] | None = None,
    ) -> Any:
        """Читает documents по списку ID."""
        return self._index.documents.fetch(
            namespace=self._resolve_namespace(namespace, require_non_empty=True),
            ids=list(ids),
            include_fields=include_fields,
        )

    def update_vector(
        self,
        id: str,
        *,
        values: Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
    ) -> Any:
        """Частично обновляет вектор."""
        kwargs: dict[str, Any] = {
            "id": id,
            "namespace": self._resolve_namespace(namespace),
        }
        if values is not None:
            kwargs["values"] = list(values)
        if metadata is not None:
            kwargs["set_metadata"] = dict(metadata)
        return self._index.update(**kwargs)

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        delete_all: bool = False,
        filter: Mapping[str, Any] | None = None,
        namespace: str | None = None,
    ) -> Any:
        """Удаляет векторы по ID, фильтру или все в namespace."""
        return self._index.delete(
            ids=list(ids) if ids is not None else None,
            delete_all=delete_all,
            filter=filter,
            namespace=self._resolve_namespace(namespace),
        )

    def delete_documents(
        self,
        *,
        ids: Sequence[str] | None = None,
        filter: Mapping[str, Any] | None = None,
        namespace: str | None = None,
    ) -> Any:
        """Удаляет documents по ID или фильтру."""
        return self._index.documents.delete(
            namespace=self._resolve_namespace(namespace, require_non_empty=True),
            ids=list(ids) if ids is not None else None,
            filter=filter,
        )

    def describe_stats(self, *, filter: Mapping[str, Any] | None = None) -> Any:
        """Возвращает статистику индекса."""
        return self._index.describe_index_stats(filter=filter)

    def close(self) -> None:
        """Закрывает соединение с индексом."""
        close = getattr(self._index, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> PineconeManager:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _manual_connection_test() -> int:
    """Ручная проверка подключения менеджера к индексу Pinecone.

    Запуск::

        python pinecone_manager.py
    """
    import sys

    print("PineconeManager connection test")
    print("-" * 40)

    try:
        with PineconeManager() as manager:
            print(f"index_name : {manager.index_name}")
            print(f"namespace  : {manager.namespace!r}")
            print(f"embedding  : {manager.embedding_model}")

            stats = manager.describe_stats()
            dimension = getattr(stats, "dimension", None)
            total = getattr(stats, "total_vector_count", None)
            metric = getattr(stats, "metric", None)
            namespaces = getattr(stats, "namespaces", None) or {}

            print(f"dimension  : {dimension}")
            print(f"metric     : {metric}")
            print(f"vectors    : {total}")
            if isinstance(namespaces, Mapping):
                print(f"namespaces : {len(namespaces)}")
                for name, info in namespaces.items():
                    count = getattr(info, "vector_count", info)
                    print(f"  - {name!r}: {count}")
            else:
                print(f"namespaces : {namespaces}")

        print("-" * 40)
        print("OK: connection to Pinecone succeeded.")
        return 0
    except Exception as exc:
        print("-" * 40)
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_manual_connection_test())
