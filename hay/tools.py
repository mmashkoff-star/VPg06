"""Инструменты агента: внешние API (собаки, погода, фото города)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ImageContent
from haystack.tools import create_tool_from_function
from haystack.utils import Secret

logger = logging.getLogger(__name__)

DOG_FACT_URL = "https://dogapi.dog/api/v2/facts"
DOG_IMAGE_URL = "https://dog.ceo/api/breeds/image/random"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_HEADERS = {
    "User-Agent": "HayTelegramAssistant/1.0 (personal assistant bot)",
    "Accept": "application/json",
}

# Коды погоды WMO → краткое описание на русском.
WMO_WEATHER: dict[int, str] = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозь / туманный туман",
    51: "лёгкая морось",
    53: "умеренная морось",
    55: "сильная морось",
    56: "лёгкая ледяная морось",
    57: "сильная ледяная морось",
    61: "небольшой дождь",
    63: "умеренный дождь",
    65: "сильный дождь",
    66: "лёгкий ледяной дождь",
    67: "сильный ледяной дождь",
    71: "небольшой снег",
    73: "умеренный снег",
    75: "сильный снег",
    77: "снежные зёрна",
    80: "небольшие ливни",
    81: "умеренные ливни",
    82: "сильные ливни",
    85: "небольшой снежный ливень",
    86: "сильный снежный ливень",
    95: "гроза",
    96: "гроза с небольшим градом",
    99: "гроза с сильным градом",
}

BREED_VISION_PROMPT = """\
Ты эксперт по породам собак. По фото определи наиболее вероятную породу
(или помесь, если уверенности нет).

Ответь на русском языке кратко, но информативно, по структуре:
1) Порода (или вероятные варианты) и короткая уверенность
2) Внешние признаки, по которым ты так решил
3) Краткая предыстория: откуда порода, для чего выводили, ключевые факты
4) Характер и для кого такая собака обычно подходит

Не выдумывай точность выше реальной. Если порода неочевидна — скажи об этом.
"""


@dataclass
class ToolSession:
    """Общее состояние между вызовами tools и Telegram-ботом."""

    last_image_url: str | None = None
    last_image_caption: str | None = None
    events: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.last_image_url = None
        self.last_image_caption = None
        self.events.clear()


def _weather_description(code: Any) -> str:
    try:
        return WMO_WEATHER.get(int(code), f"код погоды {code}")
    except (TypeError, ValueError):
        return "нет данных"


def _fetch_city_photo(city_name: str, country: str | None = None) -> str | None:
    """Ищет фото города через Wikipedia REST API (бесплатно, без ключа)."""
    titles: list[str] = [city_name]
    if country:
        titles.append(f"{city_name}, {country}")
        titles.append(f"{city_name} ({country})")

    for lang in ("ru", "en"):
        for title in titles:
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
            try:
                response = requests.get(url, headers=HTTP_HEADERS, timeout=15)
            except requests.RequestException:
                logger.exception("Wikipedia request failed for title=%r", title)
                continue
            if response.status_code != 200:
                continue
            payload = response.json()
            original = (payload.get("originalimage") or {}).get("source")
            if isinstance(original, str) and original.startswith("http"):
                return original.split("?", 1)[0]
            thumbnail = payload.get("thumbnail") or {}
            source = thumbnail.get("source")
            if isinstance(source, str) and source.startswith("http"):
                clean = source.split("?", 1)[0]
                # Увеличиваем превью, если в пути есть ширина вида /330px-
                parts = clean.rsplit("/", 1)
                if len(parts) == 2 and "px-" in parts[1]:
                    suffix = parts[1].split("px-", 1)[1]
                    return f"{parts[0]}/800px-{suffix}"
                return clean
    return None


def _format_weather_report(
    *,
    city_label: str,
    current: dict[str, Any],
    units: dict[str, Any],
    photo_url: str | None,
) -> str:
    temp = current.get("temperature_2m")
    feels = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")
    description = _weather_description(current.get("weather_code"))
    observed = current.get("time")

    temp_unit = units.get("temperature_2m", "°C")
    wind_unit = units.get("wind_speed_10m", "km/h")

    lines = [
        f"Погода в городе {city_label}:",
        f"- Сейчас: {description}",
        f"- Температура: {temp}{temp_unit}",
        f"- Ощущается как: {feels}{temp_unit}",
        f"- Влажность: {humidity}%",
        f"- Ветер: {wind} {wind_unit}",
    ]
    if observed:
        lines.append(f"- Время наблюдения: {observed}")
    if photo_url:
        lines.append(f"- Фото города: {photo_url}")
        lines.append(
            "Передай пользователю сводку погоды на его языке и упомяни, "
            "что фото города будет отправлено отдельным сообщением в Telegram."
        )
    else:
        lines.append("Фото города найти не удалось.")
    return "\n".join(lines)


def _openai_secret() -> Secret:
    if os.getenv("OPENAI_API_KEY"):
        return Secret.from_env_var("OPENAI_API_KEY")
    if os.getenv("PROXYAPI_API_KEY"):
        return Secret.from_env_var("PROXYAPI_API_KEY")
    raise ValueError("OPENAI_API_KEY or PROXYAPI_API_KEY is not set")


def _chat_model() -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return model.removeprefix("openai/")


def _build_vision_generator() -> OpenAIChatGenerator:
    load_dotenv()
    return OpenAIChatGenerator(
        api_key=_openai_secret(),
        model=_chat_model(),
        api_base_url=os.getenv("OPENAI_BASE_URL") or None,
        generation_kwargs={"temperature": 0.3},
    )


def create_assistant_tools(session: ToolSession) -> list[Any]:
    """Создаёт набор tools для Haystack Agent."""

    def get_random_dog_fact() -> str:
        """Получить случайный интересный факт о собаках из бесплатного внешнего API."""
        response = requests.get(DOG_FACT_URL, timeout=15)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if not data:
            return "Не удалось получить факт о собаках: API вернул пустой ответ."
        body = (
            data[0].get("attributes", {}).get("body")
            if isinstance(data[0], dict)
            else None
        )
        if not isinstance(body, str) or not body.strip():
            return "Не удалось разобрать факт о собаках из ответа API."
        fact = body.strip()
        session.events.append("dog_fact")
        logger.info("Dog fact fetched: %s", fact if len(fact) <= 120 else fact[:120] + "...")
        return f"Случайный факт о собаках: {fact}"

    def get_random_dog_image_and_describe() -> str:
        """Получить случайную картинку собаки, загрузить её в OpenAI Vision и описать породу с краткой предысторией."""
        image_response = requests.get(DOG_IMAGE_URL, timeout=15)
        image_response.raise_for_status()
        image_payload = image_response.json()
        image_url = image_payload.get("message")
        if not isinstance(image_url, str) or not image_url.startswith("http"):
            return "Не удалось получить URL картинки собаки из Dog CEO API."

        session.last_image_url = image_url
        session.last_image_caption = "Случайная собака 🐶"
        session.events.append("dog_image")
        logger.info("Dog image fetched: %s", image_url)

        image = ImageContent.from_url(image_url, detail="low")
        vision = _build_vision_generator()
        result = vision.run(
            messages=[
                ChatMessage.from_system(BREED_VISION_PROMPT),
                ChatMessage.from_user(
                    content_parts=[
                        "Опиши породу собаки на этом фото.",
                        image,
                    ]
                ),
            ]
        )
        replies = result.get("replies") or []
        description = (replies[0].text or "").strip() if replies else ""
        if not description:
            description = "Не удалось получить описание породы от модели."

        return (
            f"Картинка собаки: {image_url}\n\n"
            f"Описание породы от vision-модели:\n{description}\n\n"
            "Передай пользователю описание и обязательно упомяни, что фото сейчас "
            "будет отправлено отдельным сообщением в Telegram."
        )

    def get_weather_and_city_photo(city: str) -> str:
        """Получить текущую погоду в указанном городе и фото этого города.

        Использует бесплатные API: Open-Meteo (геокодинг и погода) и Wikipedia (фото).
        Аргумент city — название города, например «Казань», «Moscow», «Paris».
        """
        city_name = " ".join(city.split()).strip()
        if not city_name:
            return "Укажите название города, например: Казань."

        geo_response = requests.get(
            GEOCODING_URL,
            params={
                "name": city_name,
                "count": 1,
                "language": "ru",
                "format": "json",
            },
            headers=HTTP_HEADERS,
            timeout=15,
        )
        geo_response.raise_for_status()
        results = (geo_response.json() or {}).get("results") or []
        if not results:
            return (
                f"Не удалось найти город «{city_name}». "
                "Уточните название или добавьте страну, например: «Париж, Франция»."
            )

        place = results[0]
        latitude = place.get("latitude")
        longitude = place.get("longitude")
        if latitude is None or longitude is None:
            return f"Для города «{city_name}» не удалось определить координаты."

        resolved_name = place.get("name") or city_name
        country = place.get("country")
        admin1 = place.get("admin1")
        label_parts = [resolved_name]
        if admin1 and admin1 != resolved_name:
            label_parts.append(str(admin1))
        if country:
            label_parts.append(str(country))
        city_label = ", ".join(label_parts)

        weather_response = requests.get(
            WEATHER_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
                "wind_speed_unit": "ms",
            },
            headers=HTTP_HEADERS,
            timeout=15,
        )
        weather_response.raise_for_status()
        weather_payload = weather_response.json() or {}
        current = weather_payload.get("current") or {}
        units = weather_payload.get("current_units") or {}
        if not current:
            return f"Не удалось получить текущую погоду для {city_label}."

        photo_url = _fetch_city_photo(resolved_name, country=country)
        session.events.append("weather")
        if photo_url:
            session.last_image_url = photo_url
            session.last_image_caption = f"{city_label}"
            session.events.append("city_photo")
            logger.info("City photo fetched for %s: %s", city_label, photo_url)
        else:
            logger.info("City photo not found for %s", city_label)

        logger.info(
            "Weather fetched for %s: temp=%s code=%s",
            city_label,
            current.get("temperature_2m"),
            current.get("weather_code"),
        )
        return _format_weather_report(
            city_label=city_label,
            current=current,
            units=units,
            photo_url=photo_url,
        )

    return [
        create_tool_from_function(get_random_dog_fact),
        create_tool_from_function(get_random_dog_image_and_describe),
        create_tool_from_function(get_weather_and_city_photo),
    ]
