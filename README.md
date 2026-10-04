# VPg06 — Telegram-бот с долговременной памятью

Персональный помощник в Telegram на `pyTelegramBotAPI`. В проекте два варианта:

1. **корневой бот** (`bot.py`) — прямой диалог через OpenAI/ProxyAPI + свой `PineconeManager`;
2. **Haystack-бот** (`hay/`) — tool-calling Agent + официальный `PineconeDocumentStore`.

## Варианты запуска

### 1. Корневой бот

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python.exe bot.py
```

### 2. Haystack-бот

Подробности — в [`hay/README.md`](hay/README.md).

```powershell
.\.venv\Scripts\python.exe -m pip install -r hay\requirements.txt
.\.venv\Scripts\python.exe hay\hay-telegram-bot.py
```

Возможности Haystack-варианта:

- Agent с tools;
- контекст пользователя в Pinecone (cosine similarity);
- в индекс пишутся только сообщения пользователя и профиль, не ответы бота;
- факт о собаках, картинка собаки + vision-описание породы;
- погода по городу (Open-Meteo) и фото города (Wikipedia);
- логи и запись переписки в `hay/logs/`.

## Структура проекта

| Путь | Назначение |
|---|---|
| `bot.py` | корневой Telegram-бот |
| `assistant.py` | LLM-диалог и память через `PineconeManager` |
| `pinecone_manager.py` | менеджер Pinecone: upsert/query/`remember` |
| `chat_logger.py` | логи и запись переписки корневого бота |
| `hay/` | Haystack Agent + PineconeDocumentStore |
| `.env.example` | пример переменных окружения |
| `requirements.txt` | зависимости корневого бота (+ Haystack-пакеты) |

## Требования

- Python 3.10+
- аккаунт [Pinecone](https://www.pinecone.io/) и dense-индекс (metric: `cosine`; для `text-embedding-3-small` — dimension `1536`)
- ключ OpenAI или ProxyAPI
- токен бота от [@BotFather](https://t.me/BotFather)

## Проверка подключения к Pinecone

Для корневого менеджера:

```powershell
python pinecone_manager.py
```

Скрипт выведет имя индекса, namespace, размерность, метрику и число векторов. При успехе — `OK`, при ошибке — `FAIL`.

## Команды корневого бота

| Команда | Описание |
|---|---|
| `/start` | начать работу; сохранить профиль, если есть имя или username |
| `/help` | справка |
| `/memory` | показать, что бот помнит |
| `/forget` | очистить долговременную память |
| `/reset` | сбросить короткий диалог (память в Pinecone сохраняется) |

Команды Haystack-бота дополнительно включают `/dogfact`, `/dogpic`, `/weather` — см. [`hay/README.md`](hay/README.md).

## Как устроена память корневого бота

1. Бот ищет в Pinecone релевантные фрагменты памяти пользователя (`user_id` в metadata/filter).
2. LLM отвечает с учётом найденной памяти и короткой истории диалога.
3. Сообщение пользователя и ответ бота записываются через `remember()`.
4. Дополнительно LLM может извлечь структурированные факты.

Перед записью считается косинусное сходство с уже сохранёнными фрагментами того же пользователя:

- **ниже порога** — новая информация → новый слот (`stored`);
- **на пороге и выше, тот же текст** — дубликат → пропуск (`skipped`);
- **на пороге и выше, другой текст** — вариация → обновление слота (`updated`).

Порог задаётся в `pinecone_manager.py`:

```python
COSINE_SIMILARITY_THRESHOLD = 0.85
```

## Переменные окружения

См. `.env.example` (и `hay/.env.example` для Haystack-варианта).

| Переменная | Обязательная | Описание |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | да | токен Telegram-бота |
| `PINECONE_API_KEY` | да | API-ключ Pinecone |
| `PINECONE_INDEX_NAME` | да | имя индекса (`PINECONE_INDEX` — синоним) |
| `PROXYAPI_API_KEY` или `OPENAI_API_KEY` | да | ключ для chat/embeddings |
| `OPENAI_BASE_URL` | нет | например `https://api.proxyapi.ru/v1` |
| `OPENAI_MODEL` | нет | модель чата |
| `OPENAI_EMBEDDING_MODEL` | нет | модель эмбеддингов |
| `PINECONE_NAMESPACE` | нет | namespace в индексе |
| `PINECONE_DIMENSION` | нет | для Haystack, по умолчанию `1536` |

Файл `.env` не должен попадать в git.

## Логи

| Путь | Содержимое |
|---|---|
| `logs/bot.log` | техлоги корневого бота |
| `logs/conversations.log` | переписка корневого бота |
| `logs/chats/<user_id>.log` | переписка по пользователю |
| `hay/logs/...` | логи и переписка Haystack-бота |

Папки `logs/` и `hay/logs/` не коммитятся.
