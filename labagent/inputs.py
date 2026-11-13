"""Данные задания: любое число файлов любого вида (--input, можно несколько раз).

  картинки (png, jpg, webp…)  → отправляются модели как изображения;
  PDF                         → текст + страницы как картинки (схема в PDF — это рисунок,
                                 из текста её не достать); скан без текста — только картинки;
  DOCX                        → текст + вставленные в документ рисунки;
  TXT / MD                    → текст.

Всё собирается в один запрос: текст задания + тексты файлов + до MAX_IMAGES картинок.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .rag import read_document

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
TEXT_EXT = {".txt", ".md"}
MAX_IMAGES = int(os.environ.get("LAB_MAX_IMAGES", "6"))   # каждая картинка — сотни токенов контекста
MAX_PDF_PAGES = 4
MAX_TEXT = 6000


class InputError(RuntimeError):
    pass


@dataclass
class TaskInput:
    text: str
    images: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)   # что и откуда взято — печатается в лог

    def prompt(self) -> str:
        if not self.images:
            return self.text
        return self.text + (f"\n\n(К заданию приложено картинок: {len(self.images)} — схема и/или "
                            "данные варианта. Внимательно сверь номиналы и соединения с картинкой.)")


def _pdf_pages(path: Path, out: Path, only: list[int] | None = None) -> list[str]:
    """Страницы PDF → PNG (only — номера нужных страниц с нуля). pypdfium2 (pip) или pdftoppm (poppler)."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        files = []
        for i in (only or range(min(len(pdf), MAX_PDF_PAGES)))[:MAX_PDF_PAGES]:
            img = pdf[i].render(scale=150 / 72).to_pil()
            f = out / f"{path.stem}-p{i + 1}.png"
            img.save(f)
            files.append(str(f))
        pdf.close()
        return files
    except ImportError:
        pass
    if shutil.which("pdftoppm"):
        first, last = (min(only) + 1, max(only) + 1) if only else (1, MAX_PDF_PAGES)
        subprocess.run(["pdftoppm", "-r", "150", "-png", "-f", str(first), "-l", str(last), str(path),
                        str(out / path.stem)], check=True, capture_output=True, timeout=120)
        return sorted(str(f) for f in out.glob(f"{path.stem}*.png"))
    raise InputError("чтобы показать модели страницы PDF, нужен pypdfium2 (pip install pypdfium2) "
                     "или poppler (sudo pacman -S poppler)")


def _docx_images(path: Path, out: Path) -> list[str]:
    files = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if name.startswith("word/media/") and Path(name).suffix.lower() in IMAGE_EXT:
                f = out / f"{path.stem}-{Path(name).name}"
                f.write_bytes(z.read(name))
                files.append(str(f))
    return files


# ---------------------------------------------------------------- выбор варианта

def _variant_re(variant: str) -> re.Pattern:
    v = re.escape(str(variant).strip())
    return re.compile(rf"(?<![0-9A-Za-zА-Яа-яЁё]){v}(?![0-9A-Za-zА-Яа-яЁё])", re.I)


def is_variant_table(text: str) -> bool:
    """Таблица вариантов: много коротких строк, начинающихся с номеров 1, 2, 3… подряд.
    (В методичке тоже есть «1. … 2. …», но пунктов мало и строки длинные.)"""
    nums = []
    for ln in text.splitlines():
        m = re.match(r"^\s*(\d+)[\s.)]\s*(.*)$", ln)
        if m and len(m.group(2)) < 60:
            nums.append(int(m.group(1)))
    distinct = sorted(set(nums))
    return len(distinct) >= 10 and distinct[:3] == [1, 2, 3]


def pick_variant_rows(text: str, variant: str) -> str | None:
    """Таблица вариантов (строки «32 LM2931CS 25») → шапка таблицы + строка нужного варианта.
    None — если это не похоже на таблицу вариантов или строки нет."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    numbered = [ln for ln in lines if re.match(r"^\d+[\s.)]", ln)]
    if not is_variant_table(text):
        return None
    rows = [ln for ln in numbered if re.match(rf"^{re.escape(str(variant).strip())}[\s.)]", ln)]
    if not rows:
        return None
    first = lines.index(numbered[0])
    header = " ".join(lines[:first])
    return f"{header}\n{rows[0]}" if header else rows[0]


def pick_variant_pages(pages: list[str], variant: str) -> list[int]:
    """Страницы PDF, где написано «Вариант 7» / «Вариант № 7» — для PDF, где варианты нарисованы."""
    rx = re.compile(rf"вариант\w*\s*№?\s*{re.escape(str(variant).strip())}(?![0-9])", re.I)
    return [i for i, t in enumerate(pages) if rx.search(t or "")]


def load_inputs(task: str, files: list[str | Path] | None = None, variant: str | None = None) -> TaskInput:
    """Текст задания + файлы → одно задание для модели.
    variant — номер варианта: из таблицы вариантов берётся только его строка, из PDF с рисунками —
    только его страница, из нескольких картинок — та, в имени которой есть номер."""
    ti = TaskInput(text=task.strip())
    if variant:
        ti.text += f"\nВариант: {variant}."
    if not files:
        return ti
    tmp = Path(tempfile.mkdtemp(prefix="lab-inputs-"))
    images: list[str] = []
    files = [Path(f).expanduser() for f in files]
    if variant:   # несколько картинок вариантов — оставить ту, где в имени номер варианта
        pics = [f for f in files if f.suffix.lower() in IMAGE_EXT]
        hit = [f for f in pics if _variant_re(variant).search(f.stem)]
        if len(pics) > 1 and hit:
            files = [f for f in files if f not in pics or f in hit]
            ti.notes.append(f"вариант {variant}: выбрана картинка {', '.join(f.name for f in hit)}")
    for f in files:
        p = Path(f).expanduser()
        if not p.is_file():
            raise InputError(f"файл задания не найден: {p}")
        ext = p.suffix.lower()
        if ext in IMAGE_EXT:
            images.append(str(p.resolve()))
            ti.notes.append(f"{p.name}: картинка")
        elif ext in TEXT_EXT:
            ti.text += f"\n\nСодержимое файла «{p.name}»:\n{p.read_text(encoding='utf-8', errors='replace')[:MAX_TEXT]}"
            ti.notes.append(f"{p.name}: текст")
        elif ext == ".pdf":
            from pypdf import PdfReader

            page_texts = [pg.extract_text() or "" for pg in PdfReader(str(p)).pages]
            text = "\n\n".join(page_texts).strip()
            row = pick_variant_rows(text, variant) if variant else None
            if row:                       # таблица вариантов: модели — только строка своего варианта
                ti.text += f"\n\nИз таблицы вариантов «{p.name}» (вариант {variant}):\n{row}"
                ti.notes.append(f"{p.name}: таблица вариантов, взята строка варианта {variant}: {row.splitlines()[-1]}")
                continue
            only = pick_variant_pages(page_texts, variant) if variant else []
            pages = _pdf_pages(p, tmp, only or None)
            images += pages
            if only:
                text = "\n\n".join(page_texts[i] for i in only).strip()
            if len(text) > 50:
                ti.text += f"\n\nТекст файла «{p.name}»:\n{text[:MAX_TEXT]}"
            ti.notes.append(f"{p.name}: PDF, текст {len(text)} символов, страниц-картинок {len(pages)}"
                            + (f" (страницы варианта {variant}: {', '.join(str(i + 1) for i in only)})" if only else "")
                            + ("" if len(text) > 50 else " (скан — модель читает его как картинки)"))
        elif ext == ".docx":
            text = read_document(p).strip()
            pics = _docx_images(p, tmp)
            images += pics
            if text:
                ti.text += f"\n\nТекст файла «{p.name}»:\n{text[:MAX_TEXT]}"
            ti.notes.append(f"{p.name}: DOCX, текст {len(text)} символов, рисунков {len(pics)}")
        else:
            raise InputError(f"«{p.name}»: формат {ext} не поддерживается "
                             "(картинка, PDF, DOCX, TXT или MD)")
    if len(images) > MAX_IMAGES:
        ti.notes.append(f"картинок {len(images)}, модели отправлены первые {MAX_IMAGES} (LAB_MAX_IMAGES)")
        images = images[:MAX_IMAGES]
    ti.images = images
    return ti


MANUAL_HINTS = re.compile(r"Программа выполнения работы|Содержание отч[её]та|Методические указания|"
                          r"Типичные ошибки", re.I)
TEXT_TYPES = {".pdf", ".docx", ".md", ".txt"}


def classify(paths: list[str | Path]) -> tuple[list[Path], list[Path]]:
    """Файлы (и папки) → (файлы задания, методички). Решается по содержимому, а не по имени:
    PDF/DOCX/TXT, где есть «Программа выполнения работы», «Содержание отчёта» или «Методические
    указания», — методичка; всё остальное (картинки вариантов, таблицы вариантов) — задание."""
    files: list[Path] = []
    for x in paths:
        x = Path(x).expanduser()
        if x.is_dir():
            files += sorted(f for f in x.iterdir() if f.is_file() and not f.name.startswith(".")
                            and f.suffix.lower() in IMAGE_EXT | TEXT_TYPES)
        else:
            files.append(x)
    task_files, manuals = [], []
    for f in files:
        if not f.is_file():
            raise InputError(f"файл не найден: {f}")
        if f.suffix.lower() in TEXT_TYPES:
            try:
                head = read_document(f)[:20000]
            except Exception:
                head = ""
            if MANUAL_HINTS.search(head) and not is_variant_table(head):
                manuals.append(f)
                continue
        task_files.append(f)
    return task_files, manuals


DEFAULT_TASK = ("Выполни лабораторную работу по приложенным материалам: задание варианта (картинка схемы "
                "или данные варианта) и методические указания.")


def prepare(task: str, files: list | None = None, manuals: list | None = None, text_only: bool = False,
            llm=None, log_fn=print, variant: str | None = None) -> tuple[str, list[str], Path | None]:
    """Всё, что подаётся на вход одной работе → (текст для модели, картинки, индекс методички).
    Постоянная база (ГОСТ, замечания) подключается отдельно — она общая для всех работ."""
    from .manual import attach_manuals

    ti = load_inputs(task or DEFAULT_TASK, files, variant)
    for n in ti.notes:
        log_fn(f"Задание: {n}")
    for m in manuals or []:
        log_fn(f"Методичка: {Path(m).name}")
    text, task_index = attach_manuals(ti.prompt(), manuals, llm, log_fn)
    images = [] if text_only else ti.images
    if text_only and ti.images:
        log_fn(f"--text-only: картинки ({len(ti.images)}) модели не отправляются")
    return text, images, task_index


def spec_files(spec: dict, base: Path) -> tuple[list[str], list[str]]:
    """Из tasks/*.json: «files» (разбираются по содержимому, как в командной строке) и старые поля
    image / inputs (задание) и manual (методичка)."""
    def rel(x):
        return str((base / x).resolve()) if x and not Path(x).is_absolute() else x

    def as_list(v):
        return [v] if isinstance(v, str) else list(v or [])

    t, m = classify([rel(x) for x in as_list(spec.get("files"))])
    files = [str(x) for x in t] + [rel(x) for x in as_list(spec.get("image")) + as_list(spec.get("inputs"))]
    return files, [str(x) for x in m] + [rel(x) for x in as_list(spec.get("manual"))]
