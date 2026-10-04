# Haystack Telegram Assistant

Персональный помощник в Telegram на **Haystack Agent** + **PineconeDocumentStore** + **PyTelegramBotAPI**.

Бот ведёт диалог как реальный помощник: учитывает контекст пользователя из Pinecone (cosine similarity), умеет вызывать внешние tools и отвечает на языке пользователя.

## Возможности

- tool-calling Agent ([tutorial 43](https://haystack.deepset.ai/tutorials/43_building_a_tool_calling_agent));
- долговременный контекст в Pinecone через официальную интеграцию ([PineconeDocumentStore](https://haystack.deepset.ai/integrations/pinecone-document-store), идея RAG из [tutorial 27](https://haystack.deepset.ai/tutorials/27_first_rag_pipeline));
- в индекс попадают **только сообщения пользователя** (и базовый профиль); ответы бота в Pinecone не пишутся;
- меню команд в Telegram;
- запись технических логов и переписки в файлы;
- tools внешних данных:
  - случайный факт о собаках — [Dog API](https://dogapi.dog/);
  - случайная картинка собаки — [Dog CEO](https://dog.ceo/dog-api/) + описание породы через OpenAI Vision;
  - погода и фото города — [Open-Meteo](https://open-meteo.com/) + [Wikipedia REST API](https://www.mediawiki.org/wiki/API:REST_API).

## Структура

| Файл | Назначение |
|---|---|
| `hay-telegram-bot.py` | точка входа Telegram-бота, команды, polling |
| `assistant.py` | Haystack `Agent`, системный промпт, диалог |
| `memory.py` | `PineconeDocumentStore` + embed/retrieve по cosine |
| `tools.py` | tools: собаки, погода, фото города |
| `chat_logger.py` | техлоги и запись переписки |
| `.env.example` | пример переменных окружения |
| `requirements.txt` | зависимости |

## Требования

- Python 3.10+
- dense-индекс [Pinecone](https://www.pinecone.io/): metric `cosine`, dimension `1536` для `text-embedding-3-small`
- ключ OpenAI или ProxyAPI (chat + embeddings + vision)
- токен бота от [@BotFather](https://t.me/BotFather)

## Установка

Из корня проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r hay\requirements.txt
copy hay\.env.example hay\.env
```

Можно использовать общий `.env` в корне проекта — бот читает и корневой, и `hay/.env`.

## Переменные окружения

См. `hay/.env.example`.

| Переменная | Обязательная | Описание |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | да | токен Telegram-бота |
| `PINECONE_API_KEY` | да | API-ключ Pinecone |
| `PINECONE_INDEX_NAME` | да | имя dense-индекса |
| `PROXYAPI_API_KEY` или `OPENAI_API_KEY` | да | ключ для chat / embeddings / vision |
| `OPENAI_BASE_URL` | нет | например `https://api.proxyapi.ru/v1` |
| `OPENAI_MODEL` | нет | модель чата, по умолчанию `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | нет | модель эмбеддингов, по умолчанию `text-embedding-3-small` |
| `PINECONE_NAMESPACE` | нет | namespace, по умолчанию `hay-assistant` |
| `PINECONE_DIMENSION` | нет | размерность векторов, по умолчанию `1536` |
| `PINECONE_CLOUD` | нет | cloud для serverless, по умолчанию `aws` |
| `PINECONE_REGION` | нет | region, по умолчанию `us-east-1` |

## Запуск

```powershell
.\.venv\Scripts\python.exe hay\hay-telegram-bot.py
```

## Команды бота

| Команда | Описание |
|---|---|
| `/start` | приветствие, сохранение профиля |
| `/help` | справка |
| `/memory` | показать релевантный контекст из Pinecone |
| `/forget` | очистить долговременный контекст пользователя |
| `/reset` | сбросить короткий диалог (контекст в Pinecone сохраняется) |
| `/dogfact` | случайный факт о собаках |
| `/dogpic` | случайная картинка собаки + описание породы |
| `/weather Казань` | текущая погода и фото города |

Обычные сообщения тоже работают: можно написать «какая погода в Париже» или «расскажи факт о собаках» — Agent сам выберет tool.

Служебные ответы `/start`, `/help`, `/memory`, `/forget`, `/reset` в файлы переписки не пишутся.

## Tools

| Tool | Источник | Что делает |
|---|---|---|
| `get_random_dog_fact` | `dogapi.dog` | случайный факт о собаках |
| `get_random_dog_image_and_describe` | `dog.ceo` + OpenAI Vision | картинка собаки и описание породы / предыстории |
| `get_weather_and_city_photo` | Open-Meteo + Wikipedia | текущая погода по городу и фото города |

Фото (собака или город) отправляется в Telegram отдельным сообщением после текстового ответа.

## Как работает контекст

1. Сообщение пользователя эмбеддится (`OpenAITextEmbedder`).
2. `PineconeEmbeddingRetriever` достаёт ближайшие документы этого `user_id` по cosine similarity.
3. Haystack Agent отвечает с учётом найденного контекста, короткой истории диалога в памяти процесса и доступных tools.
4. В `PineconeDocumentStore` сохраняется только текст пользователя (`role=user`) и при `/start` — профиль (`role=profile`).
5. Ответы бота в индекс не записываются: они остаются только в короткой истории сессии и в файлах логов переписки.

Что хранится в Pinecone:

| meta.role | Что это |
|---|---|
| `user` | сообщения пользователя |
| `profile` | имя / username из Telegram |

Что не хранится: ответы ассистента, служебные ответы команд, результаты tools.

## Логи

| Путь | Содержимое |
|---|---|
| `hay/logs/bot.log` | технические логи: входящие, ответы, ошибки, tools |
| `hay/logs/conversations.log` | сохранённая переписка |
| `hay/logs/chats/<user_id>.log` | переписка по пользователю |

Папка `hay/logs/` не коммитится.
