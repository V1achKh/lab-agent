"""Проверка готового отчёта по списку замечаний преподавателя.

Список — data/docs/zamechaniya_prepodavatelya.md (постоянная часть базы знаний, его
пишет сам студент: за что преподаватель снижал оценку). Он используется дважды:
  1. до работы — добавляется к системному промпту агента («вот за что ставят минус»);
  2. после работы — отчёт проверяется:
       • автоматически, кодом: незаполненный титульник, nan/None, точка вместо запятой,
         «21.2500000», пустые разделы, нет вывода;
       • моделью: каждый пункт списка против текста отчёта (то, что кодом не проверить).
     Результат — review.md рядом с отчётом.

Проверять можно и отчёт, написанный руками: python -m labagent review мой_отчёт.docx
"""
from __future__ import annotations

import re
from pathlib import Path

from . import config

CHECKLIST = config.DOCS_DIR / "zamechaniya_prepodavatelya.md"
MAX_REPORT_CHARS = 14000

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def checklist_items(path: Path | None = None) -> str:
    """Только сами пункты («- …») и заголовки групп — без пояснений для студента."""
    path = Path(path or CHECKLIST)
    if not path.exists():
        return ""
    keep = [ln for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.startswith("- ") or ln.startswith("## ")]
    return "\n".join(keep).strip()


def with_checklist(system: str) -> str:
    items = checklist_items()
    if not items:
        return system
    return (system + "\n\nЗА ЧТО ПРЕПОДАВАТЕЛЬ СНИЖАЕТ ОЦЕНКУ — отчёт должен соблюдать каждый пункт:\n"
            + items[:4000])


# ---------------------------------------------------------------- чтение отчёта

def _text(el) -> str:
    """Текст элемента; у формул — строки системы через «; », дроби как (a)/(b)."""
    if el.tag in (W + "t", M + "t"):
        return el.text or ""
    if el.tag == W + "r":   # нижний индекс в тексте Word → «R_2», чтобы проверяющая модель его «видела»
        va = el.find(f"{W}rPr/{W}vertAlign")
        inner = "".join(_text(c) for c in el)
        return f"_{inner}" if va is not None and va.get(W + "val") == "subscript" and inner else inner
    if el.tag in (W + "br", W + "cr"):
        return "\n"
    if el.tag == W + "tab":
        return " "
    if el.tag == M + "eqArr":
        return "; ".join(_text(e) for e in el if e.tag == M + "e")
    if el.tag == M + "f":
        num, den = el.find(M + "num"), el.find(M + "den")
        return f"({_text(num) if num is not None else ''})/({_text(den) if den is not None else ''})"
    return "".join(_text(c) for c in el)


def _paragraph(p) -> dict:
    texts = [_text(p)]
    body, bold_all, has_text = [], True, False
    for el in p.iter(W + "t"):
        body.append(el.text or "")
    for r in p.iter(W + "r"):
        t = "".join(x.text or "" for x in r.iter(W + "t"))
        if not t.strip():
            continue
        has_text = True
        rpr = r.find(W + "rPr")
        b = rpr.find(W + "b") if rpr is not None else None
        if b is None or b.get(W + "val") in ("0", "false"):
            bold_all = False
    return {"text": "".join(texts).strip(), "body": "".join(body).strip(),
            "math": p.find(".//" + M + "oMath") is not None,
            "figure": p.find(".//" + W + "drawing") is not None,
            "heading": has_text and bold_all}


def read_report(path: str | Path) -> list[dict]:
    """Абзацы и таблицы отчёта по порядку (формулы Word — тоже текстом)."""
    from docx import Document

    doc = Document(str(path))
    out = []
    for el in doc.element.body:
        if el.tag == W + "p":
            item = _paragraph(el)
            if item["text"] or item["figure"]:
                out.append(item)
        elif el.tag == W + "tbl":
            rows = []
            for tr in el.iter(W + "tr"):
                cells = ["".join(t.text or "" for t in tc.iter(W + "t")).strip() for tc in tr.iter(W + "tc")]
                rows.append(" | ".join(cells))
            text = "\n".join(rows)
            out.append({"text": text, "body": text, "math": False, "figure": False, "heading": False,
                        "table": True})
    return out


def as_text(items: list[dict]) -> str:
    lines = []
    for it in items:
        if it["figure"] and not it["text"]:
            lines.append("[рисунок]")
        elif it.get("table"):
            lines.append("[таблица]\n" + it["text"])
        elif it["heading"]:
            lines.append(f"\n## {it['text']}")
        else:
            lines.append(it["text"])
    return "\n".join(lines)


# ---------------------------------------------------------------- проверки кодом

_POINT = re.compile(r"(?<![\w.,])\d+\.\d+(?![\d.]*\.\d)")
_COMMA = re.compile(r"(?<![\w.,])\d+,\d+(?![\d,]*,\d)")
_LONG = re.compile(r"\d+[.,]\d{10,}")   # «0.30000000000000004» — мусор вычислений, а не измерение
# терминология: (неверно, как правильно)
_TERMS = [(re.compile(r"закон\w*\s+Кирхгоф", re.I), "«правила Кирхгофа», а не «законы»"),
          (re.compile(r"абсолютн\w*\s+погрешност\w*", re.I),
           "не употреблять «абсолютная погрешность» (говорить «расхождение», «погрешность измерения»)")]
_STAR = re.compile(r"[\w)]\s*\*\s*[\w(]")
# в описании схемы нельзя: узлы, стороны рисунка, номера выводов; прибор — не часть схемы
_DESCR_BAD = re.compile(r"\b(узл\w*|узел|слева|справа|сверху|снизу|верхн\w*|нижн\w*|лев\w+ (?:клемм|вывод)\w*|"
                        r"прав\w+ (?:клемм|вывод)\w*|перв\w+ (?:клемм|вывод)\w*|втор\w+ (?:клемм|вывод)\w*|"
                        r"клемм\w* №|вывод\w* №|вольтметр\w*|амперметр\w*|на рисунке|на схеме)\b", re.I)
FINAL_PHRASE = "Расхождение теоретических и измеренных значений"
_FILENAME = re.compile(r"^ИУ\d+-\S+ \S+ Лаб\.\s?\d+", re.I)
_BAD = re.compile(r"\b(nan|NaN|inf|None|null|TODO)\b|\{|\}|\?\?\?")
_SKIP_POINT = ("ГОСТ", "Multisim", "версия", "ver", "Python")


def _numbers(text: str, rx) -> list[str]:
    out = []
    for m in rx.finditer(text):
        before = text[max(0, m.start() - 12):m.start()]
        if not any(sk in before for sk in _SKIP_POINT):
            out.append(m.group(0))
    return out


def auto_checks(items: list[dict], path: str | Path | None = None) -> list[str]:
    """Проверки кодом по списку преподавателя (то, что можно проверить без модели)."""
    out = []
    full = "\n".join(it["text"] for it in items)
    for ph in ("Фамилия И. О.", "____"):
        if ph in full:
            out.append(f"Не заполнены данные титульного листа: «{ph}» (впишите их в .env: LAB_STUDENT, LAB_GROUP)")
    if path is not None and not _FILENAME.match(Path(path).stem):
        out.append(f"Имя файла «{Path(path).name}» — нужно «ИУ5-3… Фамилия Лаб.1»")
    # десятичный разделитель — любой, но один на весь отчёт
    points, commas = _numbers(full, _POINT), _numbers(full, _COMMA)
    if points and commas:
        out.append(f"Десятичный разделитель разный: «{points[0]}» и «{commas[0]}» — нужен один на весь отчёт")
    section = ""
    descr: list[dict] = []
    for it in items:
        if it["heading"]:
            section = it["text"].lower()
            continue
        if "описание схемы" in section:
            descr.append(it)
        body = it["body"]
        m = _BAD.search(body)
        if m:
            out.append(f"Служебное значение «{m.group(0)}» в тексте: «{body[:80]}»")
        for rx, hint in _TERMS:
            m = rx.search(it["text"])
            if m:
                out.append(f"Терминология: «{m.group(0)}» — нужно {hint}")
        if _STAR.search(it["text"]):
            out.append(f"Знак «*» как умножение: «{it['text'][:60]}» — нужен «·» или «×»")
        m = _LONG.search(it["text"])
        if m:
            out.append(f"Число «{m.group(0)}» похоже на неокруглённый результат вычислений")
    # описание схемы и итоговая фраза — требования «Типичных ошибок … Лаб.1»
    lab1 = bool(re.search(r"лабораторной работе\s*№\s*1\b", full))
    for it in descr if lab1 else []:
        m = _DESCR_BAD.search(it["text"])
        if m:
            out.append(f"Описание схемы: «{m.group(0)}» — в описании нельзя упоминать узлы, стороны рисунка, "
                       f"номера выводов и измерительный прибор: «{it['text'][:80]}»")
    if lab1 and descr and not re.search(r"требуется определить", descr[-1]["text"], re.I):
        out.append("Описание схемы должно заканчиваться словами «Требуется определить …»")
    for k, it in enumerate(items):
        if it["heading"] and len(it["text"]) < 120:
            nxt = next((x for x in items[k + 1:]), None)
            if nxt is None or nxt["heading"]:
                out.append(f"Пустой раздел «{it['text']}»")
    if not any("вывод" in it["text"].lower() for it in items if it["heading"]):
        out.append("Нет раздела «Вывод»")
    last = next((it for it in reversed(items) if not it["heading"] and it["text"]), None)
    if lab1 and (last is None or FINAL_PHRASE.lower() not in last["text"].lower()):
        out.append(f"Отчёт должен заканчиваться фразой «{FINAL_PHRASE} не превосходит (приблизительно равно) "
                   "теоретически допустимую(-ой) погрешность(-и)»")
    # одно и то же замечание много раз — шум
    seen, uniq = set(), []
    for r in out:
        key = r.split(":")[0]
        if r not in seen and sum(1 for u in uniq if u.split(":")[0] == key) < 3:
            seen.add(r)
            uniq.append(r)
    return uniq


# ---------------------------------------------------------------- проверка моделью

REVIEW_SYSTEM = """\
Ты — строгий преподаватель электротехники и проверяешь отчёт студента по лабораторной работе.
Тебе дан список требований (за что снижается оценка) и текст отчёта (формулы — текстом,
рисунки обозначены [рисунок]). Проверь отчёт по КАЖДОМУ пункту списка.
Указывай только реальные нарушения, которые видны в тексте, с конкретным местом.
Не придумывай нарушений; если всё соблюдено — пустой список.
Схему на рисунке ты не видишь — пункты про рисунок не оценивай.
Нижние индексы в тексте показаны знаком «_»: «R_2» — это R₂ с индексом (так и должно быть).
Слова «клемма», «вывод», «провод», «точка соединения» в описании схемы разрешены (запрещены узлы,
стороны рисунка и НОМЕРА выводов). Не считай нарушением то, чего пункт не требует.
Ответ — только JSON:
{"remarks": [{"rule": "пункт списка", "where": "раздел или цитата", "comment": "что не так и как исправить"}]}"""


def llm_review(text: str, llm) -> list[dict]:
    from .llm import extract_json

    items = checklist_items()
    if not items:
        return []
    user = f"Требования:\n{items}\n\nОтчёт:\n<<<\n{text[:MAX_REPORT_CHARS]}\n>>>"
    # ответ — короткий список нарушений; предел длины, чтобы зациклившаяся модель не писала минутами
    data = extract_json(llm.ask(REVIEW_SYSTEM, user, json_mode=True, max_tokens=1500))
    out = []
    for r in data.get("remarks", []) if isinstance(data, dict) else []:
        if isinstance(r, dict) and (r.get("comment") or r.get("rule")):
            out.append({k: str(r.get(k, "")) for k in ("rule", "where", "comment")})
    return out


def review_report(path: str | Path, llm=None, log=print) -> dict:
    path = Path(path)
    items = read_report(path)
    res = {"report": str(path), "auto": auto_checks(items, path), "llm": None, "llm_error": None,
           "checklist": CHECKLIST.exists()}
    if llm is not None and res["checklist"]:
        log("\nПроверка отчёта моделью по списку замечаний преподавателя (до ~1 мин)…")
        try:
            res["llm"] = llm_review(as_text(items), llm)
        except Exception as exc:  # модель недоступна или ответила не JSON — остаются проверки кодом
            res["llm_error"] = str(exc)
    lines = [f"# Проверка отчёта {path.name}", "", "## Автоматические проверки", ""]
    lines += [f"- {r}" for r in res["auto"]] or ["- замечаний нет"]
    lines += ["", "## Проверка по списку замечаний преподавателя (модель)", ""]
    if res["llm"] is None:
        lines.append(f"- не выполнялась{': ' + res['llm_error'] if res['llm_error'] else ''}")
    elif not res["llm"]:
        lines.append("- замечаний нет")
    else:
        lines += [f"- **{r['rule']}** — {r['where']}: {r['comment']}" for r in res["llm"]]
    md = path.with_name(path.stem + "_review.md")
    try:
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        res["md"] = str(md)
    except OSError:
        res["md"] = None
    return res


def print_review(res: dict, log=print) -> None:
    log(f"\nПроверка отчёта по списку замечаний ({Path(res['report']).name}):")
    if not res["checklist"]:
        log("  (нет файла data/docs/zamechaniya_prepodavatelya.md — только автоматические проверки)")
    for r in res["auto"]:
        log(f"  [авто]   {r}")
    for r in res["llm"] or []:
        log(f"  [модель] {r['rule']} — {r['where']}: {r['comment']}")
    if res.get("llm_error"):
        log(f"  проверка моделью не удалась: {res['llm_error']}")
    if not res["auto"] and not res["llm"]:
        log("  замечаний нет")
    if res.get("md"):
        log(f"  Подробно: {res['md']}")
