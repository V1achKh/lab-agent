"""Настройки. Всё берётся из переменных окружения или файла .env в корне проекта.

Пример .env:
    LAB_LLM_URL=http://localhost:8080/v1
    LAB_LLM_MODEL=gemma4-12b
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Мини-парсер .env (без внешних зависимостей).
    Как в python-dotenv: при повторе ключа побеждает последнее значение в файле;
    « # комментарий» в конце строки отбрасывается; переменные окружения важнее файла."""
    if not path.exists():
        return
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in ("'", '"') and value[-1:] == value[:1]:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key.strip()] = value
    for key, value in values.items():
        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --- Языковая модель (чат): Gemma 4 12B в llama-server (scripts/llamacpp.sh). Подойдёт и любой
#     другой OpenAI-совместимый сервер.
LLM_URL = _env("LAB_LLM_URL", "http://localhost:8080/v1")
LLM_MODEL = _env("LAB_LLM_MODEL", "gemma4-12b")
LLM_API_KEY = _env("LAB_LLM_KEY", "local")
LLM_TIMEOUT = float(_env("LAB_LLM_TIMEOUT", "900"))
LLM_TEMPERATURE = float(_env("LAB_LLM_TEMPERATURE", "0.2"))
# предел длины одного ответа модели: маленькие модели иногда «зацикливаются» и пишут без конца
LLM_MAX_TOKENS = int(_env("LAB_LLM_MAX_TOKENS", "4096"))
# Дополнительные поля запроса, например отключить «размышления» модели в llama-server:
#   LAB_LLM_EXTRA={"chat_template_kwargs": {"enable_thinking": false}}
LLM_EXTRA: dict = json.loads(_env("LAB_LLM_EXTRA", "{}"))

# --- Модель эмбеддингов для RAG: bge-m3 (хорошо понимает русский), второй llama-server на процессоре
EMBED_URL = _env("LAB_EMBED_URL", "http://localhost:8081/v1")
EMBED_MODEL = _env("LAB_EMBED_MODEL", "bge-m3")

# --- Папки
DOCS_DIR = Path(_env("LAB_DOCS_DIR", str(ROOT / "data" / "docs")))
INDEX_DIR = Path(_env("LAB_INDEX_DIR", str(ROOT / "data" / "index")))
RUNS_DIR = Path(_env("LAB_RUNS_DIR", str(ROOT / "runs")))

# --- Симуляция: auto (ngspice, если установлен, иначе встроенный решатель) | ngspice | builtin
SIM_ENGINE = _env("LAB_SIM_ENGINE", "auto")
NGSPICE_BIN = _env("LAB_NGSPICE", "ngspice")
SOFFICE_BIN = _env("LAB_SOFFICE", "soffice")

# --- Оформление отчёта: bmstu (как в отчётах группы) | gost (ГОСТ 7.32)
REPORT_STYLE = _env("LAB_REPORT_STYLE", "bmstu")
# --- Подписи на схеме: bmstu («R₁ = 1 кОм», как в заданиях) | eskd (обозначение и номинал отдельно)
SCHEMATIC_LABELS = _env("LAB_SCHEMATIC_LABELS", "bmstu")
# --- Сколько значащих цифр показывает прибор (как индикатор мультиметра в Multisim)
METER_DIGITS = int(_env("LAB_METER_DIGITS", "4"))
# --- Десятичный разделитель в отчёте, на схеме и графиках: "." (21.25 В) или "," (21,25 В).
#     Должен совпадать с правилом в data/docs/zamechaniya_prepodavatelya.md
DECIMAL = _env("LAB_DECIMAL", ",").strip() or ","
if DECIMAL not in (".", ","):
    raise SystemExit(f"LAB_DECIMAL должен быть «.» или «,», а не «{DECIMAL}»")


def dsep(text: str) -> str:
    """Подставляет выбранный десятичный разделитель в число, записанное с точкой."""
    return text.replace(".", DECIMAL)

# --- Данные для титульного листа
TITLE_ORG = _env(
    "LAB_ORG",
    "Министерство науки и высшего образования Российской Федерации\n"
    "Название университета",
)
TITLE_UNIVERSITY = _env("LAB_UNIVERSITY", "МГТУ им. Н.Э. Баумана")
TITLE_DEPT = _env("LAB_DEPT", "Кафедра электротехники")
TITLE_DISCIPLINE = _env("LAB_DISCIPLINE", "Электротехника")
TITLE_STUDENT = _env("LAB_STUDENT", "Фамилия И. О.")
TITLE_GROUP = _env("LAB_GROUP", "____")
TITLE_TEACHER = _env("LAB_TEACHER", "Фамилия И. О.")
TITLE_CITY = _env("LAB_CITY", "Город")
