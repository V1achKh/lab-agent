"""Отчёт по лабораторной в .docx и .pdf. Два стиля оформления (LAB_REPORT_STYLE):

  bmstu (по умолчанию) — как в отчётах вашей группы по курсу «Электротехника»:
      титульный лист «МГТУ им. Н.Э. Баумана / Отчёт по лабораторной работе №… / Тема / Вариант»,
      руководитель и студент справа, год внизу; дальше — полужирные подзаголовки с двоеточием
      («Полученное задание:», «Описание схемы:»…), одинарный интервал, без подписей рисунков.
  gost — ГОСТ 7.32-2017: нумерованные разделы, полуторный интервал, «Рисунок N — …»,
      «Таблица N — …», номера страниц.

Модель НЕ форматирует документ: она даёт только текст, а всё оформление задано здесь.
Формулы вставляются как настоящие формулы Word (OMML, см. omml.py).
"""
from __future__ import annotations

import datetime as dt
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt

from . import config
from .omml import add_math_paragraph

FONT = "Times New Roman"
C, L, R, J = WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.JUSTIFY


@dataclass
class ReportData:
    lab_number: str
    lab_title: str                      # тема работы
    goal: str = ""
    theory: str = ""
    equipment: str = ""
    procedure: str = ""
    conclusions: str = ""
    variant: str = ""
    # блоки: ("heading", text) | ("bold", text) | ("text", text) | ("figure", path, caption)
    #        | ("table", dict) | ("math", parts)
    blocks: list = field(default_factory=list)
    student: str = field(default_factory=lambda: config.TITLE_STUDENT)
    group: str = field(default_factory=lambda: config.TITLE_GROUP)
    teacher: str = field(default_factory=lambda: config.TITLE_TEACHER)
    discipline: str = field(default_factory=lambda: config.TITLE_DISCIPLINE)
    style: str = field(default_factory=lambda: config.REPORT_STYLE)
    # если True — разделы «Цель/Теория/Выводы» не добавляются автоматически (они уже в blocks)
    blocks_only: bool = False


# ---------------------------------------------------------------- низкоуровневое

def _set_font(run, size=14, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), FONT)


# индексы переменных в тексте (требование преподавателя: «не R2, а R₂, не Uвх, а U_ВХ»):
# R1, E2, J1, I0, U0, Δ0, φ3, RV, RA, Eдоп, Uвх → буква + нижний индекс. Обозначения приборов
# по ЕСКД (PV1, PA1) не трогаем: перед V/A стоит буква P.
_SUBS = re.compile(r"(?<![\wА-Яа-яЁё])([RCLEJIUφΔ])(\d+|V|A|доп|вх|ВХ|вых|ВЫХ|изм|ИЗМ|хх|кз)(?![\wА-Яа-яЁё])")


def _add_text_runs(p, line: str, size, bold, italic):
    pos = 0
    for m in _SUBS.finditer(line):
        if m.start() > pos:
            _set_font(p.add_run(line[pos:m.start()]), size, bold, italic)
        _set_font(p.add_run(m.group(1)), size, bold, italic)
        sub = p.add_run(m.group(2).upper() if m.group(2) in ("вх", "вых", "изм", "хх", "кз") else m.group(2))
        _set_font(sub, size, bold, italic)
        sub.font.subscript = True
        pos = m.end()
    if pos < len(line):
        _set_font(p.add_run(line[pos:]), size, bold, italic)


def _cell_borders(table):
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    table._tbl.tblPr.append(borders)


class _Writer:
    def __init__(self, style: str):
        self.style = style if style in ("bmstu", "gost") else "bmstu"
        self.gost = self.style == "gost"
        self.doc = Document()
        self.section_no = self.figure_no = self.table_no = 0
        self._first_content = True
        self._setup()

    # ---- оформление страницы и стиля «Обычный»
    def _setup(self):
        st = self.doc.styles["Normal"]
        st.font.name = FONT
        st.font.size = Pt(14)
        rpr = st.element.get_or_add_rPr()
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            rpr.append(fonts)
        for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            fonts.set(qn(attr), FONT)
        pf = st.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE if self.gost else WD_LINE_SPACING.SINGLE
        pf.first_line_indent = Cm(1.25 if self.gost else 0.5)
        pf.space_before = pf.space_after = Pt(0)
        pf.alignment = J
        pf.widow_control = True
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Mm(210), Mm(297)
        sec.left_margin, sec.right_margin = Mm(30), Mm(15)
        sec.top_margin, sec.bottom_margin = Mm(20), Mm(20)
        sec.footer_distance = Mm(10)
        if self.gost:
            self._page_numbers(sec)

    def _page_numbers(self, section):
        section.different_first_page_header_footer = True
        p = section.footer.paragraphs[0]
        p.alignment = C
        p.paragraph_format.first_line_indent = Cm(0)
        run = p.add_run()
        _set_font(run, 12)
        for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if tag:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tag)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            run._element.append(el)

    def para(self, text="", *, align=None, indent=True, size=14, bold=False, italic=False,
             keep_next=False, single=False, space_before=0, space_after=0, subs=False):
        p = self.doc.add_paragraph()
        if align is not None:
            p.alignment = align
        if not indent:
            p.paragraph_format.first_line_indent = Cm(0)
        if keep_next:
            p.paragraph_format.keep_with_next = True
        if single:
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        if space_before:
            p.paragraph_format.space_before = Pt(space_before)
        if space_after:
            p.paragraph_format.space_after = Pt(space_after)
        if text:
            lines = text.split("\n")
            for k, line in enumerate(lines):
                if subs:
                    _add_text_runs(p, line, size, bold, italic)
                    run = p.runs[-1] if p.runs else p.add_run("")
                else:
                    run = p.add_run(line)
                    _set_font(run, size, bold, italic)
                if k < len(lines) - 1:
                    run.add_break()
        return p

    def _new_page_if_first(self, p):
        if self._first_content:
            p.paragraph_format.page_break_before = True
            self._first_content = False

    # ---- титульные листы
    def title_bmstu(self, d: ReportData):
        year = dt.date.today().year
        self.para(config.TITLE_UNIVERSITY, align=C, indent=False)
        for _ in range(11):
            self.para(indent=False)
        self.para(f"Отчёт по лабораторной работе №{d.lab_number}\nпо курсу «{d.discipline}»", align=C, indent=False)
        self.para(f"Тема: {d.lab_title.rstrip('.')}.", align=C, indent=False)
        if d.variant:
            self.para(f"Вариант {d.variant}.", align=C, indent=False)
        for _ in range(9):
            self.para(indent=False)
        self.para(f"Руководитель\n{d.teacher}", align=R, indent=False)
        self.para(indent=False)
        self.para(f"Студент группы {d.group}\n{d.student}", align=R, indent=False)
        for _ in range(13):
            self.para(indent=False)
        self.para(f"{year} г.", align=C, indent=False)

    def title_gost(self, d: ReportData):
        org = config.TITLE_ORG.replace("\\n", "\n")
        for line in org.split("\n"):
            self.para(line, align=C, indent=False, single=True)
        self.para(config.TITLE_DEPT, align=C, indent=False, single=True)
        for _ in range(6):
            self.para(indent=False, single=True)
        self.para("ОТЧЁТ", align=C, indent=False, bold=True, size=16)
        self.para(f"по лабораторной работе № {d.lab_number}", align=C, indent=False)
        self.para(f"«{d.lab_title}»", align=C, indent=False, bold=True)
        self.para(f"по дисциплине «{d.discipline}»" + (f", вариант {d.variant}" if d.variant else ""),
                  align=C, indent=False)
        for _ in range(5):
            self.para(indent=False, single=True)
        tbl = self.doc.add_table(rows=2, cols=3)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        tbl._tbl.tblPr.append(layout)
        widths = (Cm(7.5), Cm(4), Cm(4.5))
        for gc, w in zip(tbl._tbl.tblGrid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(w.twips)))
        cells = [(f"Выполнил студент гр. {d.group}", "", d.student), ("Проверил", "", d.teacher)]
        for r, row in enumerate(cells):
            for c, text in enumerate(row):
                cell = tbl.cell(r, c)
                cell.width = widths[c]
                p = cell.paragraphs[0]
                p.paragraph_format.first_line_indent = Cm(0)
                p.alignment = L if c != 1 else C
                _set_font(p.add_run(text or "_____________"), 14)
        for _ in range(5):
            self.para(indent=False, single=True)
        self.para(f"{config.TITLE_CITY} {dt.date.today().year}", align=C, indent=False)

    # ---- содержимое
    def heading(self, text: str):
        if self.gost:
            self.section_no += 1
            p = self.para(f"{self.section_no} {text.strip().rstrip('.:')}", bold=True, keep_next=True,
                          space_before=12, space_after=6, align=L)
        else:
            p = self.para(text.strip().rstrip(".:") + ":", bold=True, keep_next=True, space_before=8, align=L)
        self._new_page_if_first(p)

    def bold(self, text: str):
        """Полужирный подзаголовок внутри раздела (в стиле bmstu — то же, что heading)."""
        if self.gost:
            p = self.para(text, bold=True, keep_next=True, space_before=6, align=L)
            self._new_page_if_first(p)
        else:
            self.heading(text)

    def text(self, text: str):
        for chunk in [t.strip() for t in text.strip().split("\n") if t.strip()]:
            p = self.para(chunk, subs=True)
            self._new_page_if_first(p)

    def math(self, parts):
        p = add_math_paragraph(self.doc, parts)
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        self._new_page_if_first(p)

    def figure(self, path: str, caption: str = "", width_cm: float = 15.0):
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
        if h * width_cm / w > 17:
            width_cm = 17 * w / h
        p = self.para(indent=False, align=C, keep_next=bool(caption and self.gost), space_before=4, space_after=4)
        self._new_page_if_first(p)
        p.add_run().add_picture(path, width=Cm(width_cm))
        if self.gost:
            self.figure_no += 1
            self.para(f"Рисунок {self.figure_no} — {(caption or 'Рисунок').strip().rstrip('.')}",
                      indent=False, align=C, space_after=6)

    def table(self, t: dict):
        if self.gost:
            self.table_no += 1
            cap = self.para(f"Таблица {self.table_no} — {t['caption'].strip().rstrip('.')}",
                            indent=False, align=L, keep_next=True, space_before=6)
            self._new_page_if_first(cap)
        headers, rows = t["headers"], t["rows"]
        table = self.doc.add_table(rows=1 + len(rows), cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        _cell_borders(table)
        for r, row in enumerate([headers] + rows):
            for c, val in enumerate(row):
                p = table.cell(r, c).paragraphs[0]
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                p.alignment = C
                _set_font(p.add_run(str(val)), 12)
                if r < len(rows):
                    p.paragraph_format.keep_with_next = True
        tr_pr = table.rows[0]._tr.get_or_add_trPr()
        hdr = OxmlElement("w:tblHeader")
        hdr.set(qn("w:val"), "true")
        tr_pr.append(hdr)
        self.para(indent=False, space_after=4)

    def block(self, b):
        kind = b[0]
        if kind == "heading":
            self.heading(b[1])
        elif kind == "bold":
            self.bold(b[1])
        elif kind == "text":
            self.text(b[1])
        elif kind == "math":
            self.math(b[1])
        elif kind == "figure":
            self.figure(b[1], b[2] if len(b) > 2 else "")
        elif kind == "table":
            self.table(b[1])


def report_filename(lab_number, ext: str = ".docx") -> str:
    """Имя файла отчёта, как требует преподаватель: «ИУ5-34Б Фамилия Лаб.1.docx»."""
    surname = (config.TITLE_STUDENT.split() or ["Фамилия"])[0]
    name = f"{config.TITLE_GROUP} {surname} Лаб.{str(lab_number).strip() or '—'}"
    return re.sub(r'[\\/:*?"<>|]+', "-", name) + ext


def build_docx(d: ReportData, out_path: Path) -> Path:
    w = _Writer(d.style)
    (w.title_gost if w.gost else w.title_bmstu)(d)
    if not d.blocks_only:
        if d.goal:
            w.heading("Цель работы")
            w.text(d.goal)
        if d.theory:
            w.heading("Краткие теоретические сведения")
            w.text(d.theory)
        if d.equipment:
            w.heading("Используемое оборудование и программы")
            w.text(d.equipment)
        if d.procedure:
            w.heading("Порядок выполнения работы")
            w.text(d.procedure)
    for b in d.blocks:
        w.block(b)
    if not d.blocks_only:
        w.heading("Выводы" if w.gost else "Вывод")
        w.text(d.conclusions or "Выводы не сформулированы.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w.doc.save(out_path)
    return out_path


def to_pdf(docx_path: Path) -> Path | None:
    """PDF через LibreOffice в фоновом режиме. Отдельный профиль, чтобы не мешать открытому офису."""
    soffice = shutil.which(config.SOFFICE_BIN) or shutil.which("libreoffice")
    if not soffice:
        return None
    docx_path = Path(docx_path).resolve()
    with tempfile.TemporaryDirectory(prefix="lo_profile_") as prof:
        cmd = [soffice, f"-env:UserInstallation=file://{prof}", "--headless", "--norestore",
               "--convert-to", "pdf", "--outdir", str(docx_path.parent), str(docx_path)]
        subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    pdf = docx_path.with_suffix(".pdf")
    return pdf if pdf.exists() else None


__all__ = ["ReportData", "build_docx", "to_pdf", "WD_BREAK"]
