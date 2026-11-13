"""Методичка, поданная на вход вместе с заданием (--manual файл, можно несколько).

База знаний делится на две части:
  • постоянная — data/docs/: ГОСТы, правила оформления, замечания преподавателя.
    Индексируется командой `python -m labagent index` и нужна любой работе;
  • методичка задания — только для этого запуска. Индексируется отдельно
    (кэш в data/index/task/<хэш>/, повторно не пересчитывается) и при поиске
    объединяется с постоянной базой. В data/docs она НЕ копируется — иначе через
    десяток лаб поиск по «лабе 3» начал бы находить куски из «лабы 5».

Кроме того, текст методички (до MAX_CHARS символов) добавляется прямо к заданию —
модель видит «Программу выполнения работы» и «Содержание отчёта» сразу, без поиска.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path

from . import config
from .rag import RagIndex, read_document

log = logging.getLogger(__name__)

MAX_CHARS = 12000          # ~4 тыс. токенов на все методички вместе: помещается даже в 8K контекста
TASK_INDEX_ENV = "LAB_TASK_INDEX"   # через эту переменную MCP-сервер узнаёт индекс методички


class ManualError(RuntimeError):
    pass


def read_manual(path: str | Path) -> str:
    path = Path(path).expanduser()
    if not path.is_file():
        raise ManualError(f"файл методички не найден: {path}")
    if path.suffix.lower() not in (".pdf", ".docx", ".md", ".txt"):
        raise ManualError(f"формат {path.suffix} не поддерживается (нужен PDF, DOCX, MD или TXT)")
    text = read_document(path).strip()
    if len(text) < 200:
        raise ManualError(
            f"из «{path.name}» почти не извлёкся текст ({len(text)} символов). Похоже, это скан без "
            "текстового слоя — сохраните методичку как PDF с текстом или DOCX (или распознайте: "
            "ocrmypdf -l rus скан.pdf текст.pdf)")
    return text


def build_task_index(paths: list[Path], llm, log_fn=print) -> Path | None:
    """Отдельный индекс для методичек этого задания. Возвращает папку индекса (или None)."""
    if not paths or llm is None:
        return None
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update(llm.embed_model.encode())
    target = config.INDEX_DIR / "task" / h.hexdigest()[:12]
    if (target / "chunks.json").exists():
        return target
    docs = target / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, docs / p.name)
    try:
        idx = RagIndex.build(llm, docs_dir=docs, index_dir=target)
    except Exception as exc:  # сервер эмбеддингов не запущен — методичка останется только в тексте задания
        shutil.rmtree(target, ignore_errors=True)
        log_fn(f"Внимание: индекс методички не построен ({exc}). Она будет только в тексте задания.")
        return None
    log_fn(f"Методичка проиндексирована отдельно от постоянной базы: {len(idx.chunks)} фрагментов")
    return target


def load_rag(task_index: Path | str | None = None) -> RagIndex | None:
    """Постоянная база + методичка задания (если есть). None — если нет ни того, ни другого."""
    try:
        base = RagIndex.load()
    except RuntimeError as exc:
        log.warning("%s", exc)
        base = None
    task = None
    task_index = task_index or os.environ.get(TASK_INDEX_ENV)
    if task_index:
        try:
            task = RagIndex.load(Path(task_index))
        except RuntimeError as exc:
            log.warning("индекс методички: %s", exc)
    return RagIndex.merge(base, task)


def manual_block(name: str, text: str, limit: int = MAX_CHARS) -> str:
    if len(text) > limit:
        text = (text[:limit] + "\n…\n(дальше методичка обрезана — остальное ищи инструментом "
                "search_docs, если он есть)")
    return f"\n\nМетодические указания к этой работе (файл «{name}»):\n<<<\n{text}\n>>>"


def attach_manuals(task: str, manuals: list[str | Path] | None, llm=None,
                   log_fn=print) -> tuple[str, Path | None]:
    """Задание + тексты методичек; второй результат — папка индекса методичек (для RAG)."""
    paths = [Path(m).expanduser().resolve() for m in (manuals or []) if m]
    if not paths:
        return task, None
    texts = [read_manual(p) for p in paths]
    limit = MAX_CHARS // len(paths)
    for p, t in zip(paths, texts):
        task += manual_block(p.name, t, limit)
    return task, build_task_index(paths, llm, log_fn)
