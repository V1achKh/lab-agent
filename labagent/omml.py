"""Формулы Word (Office Math, OMML) без внешних библиотек.

Формула описывается списком «кусочков»:
    "текст"                    обычный текст (+, =, числа)
    Sym("I", "R1")             переменная с индексом: I_R1
    Frac(num, den)             дробь (num и den — такие же списки)
    System([eq1, eq2, ...])    система уравнений с фигурной скобкой
Word и LibreOffice показывают это как настоящие редактируемые формулы.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from xml.sax.saxutils import escape

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


@dataclass
class Sym:
    base: str
    sub: str = ""


@dataclass
class Frac:
    num: list
    den: list


@dataclass
class System:
    rows: list = field(default_factory=list)


_CTRL = ('<m:ctrlPr><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>'
         '<w:sz w:val="28"/></w:rPr></m:ctrlPr>')


def _run(text: str, italic: bool = False) -> str:
    sty = "" if italic else '<m:rPr><m:sty m:val="p"/></m:rPr>'
    rpr = ('<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>'
           '<w:sz w:val="28"/></w:rPr>')
    return f'<m:r>{sty}{rpr}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'


def _expr(parts) -> str:
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append(_run(p))
        elif isinstance(p, Sym):
            if p.sub:
                out.append(f"<m:sSub><m:e>{_run(p.base, True)}</m:e><m:sub>{_run(p.sub)}</m:sub></m:sSub>")
            else:
                out.append(_run(p.base, True))
        elif isinstance(p, Frac):
            out.append(f"<m:f><m:num>{_expr(p.num)}</m:num><m:den>{_expr(p.den)}</m:den></m:f>")
        elif isinstance(p, System):
            rows = "".join(f"<m:e>{_expr(r)}</m:e>" for r in p.rows)
            out.append(f'<m:d><m:dPr><m:begChr m:val="{{"/><m:endChr m:val=""/>{_CTRL}</m:dPr>'
                       f"<m:e><m:eqArr><m:eqArrPr>{_CTRL}</m:eqArrPr>{rows}</m:eqArr></m:e></m:d>")
        else:
            raise TypeError(p)
    return "".join(out)


def omath_xml(parts, align: str = "center") -> str:
    """Формула «отдельной строкой» (oMathPara), как её вставляет сам Word."""
    return (f'<m:oMathPara xmlns:m="{M}" xmlns:w="{W}"><m:oMathParaPr><m:jc m:val="{align}"/></m:oMathParaPr>'
            f"<m:oMath>{_expr(parts)}</m:oMath></m:oMathPara>")


def add_math_paragraph(doc, parts, align_center: bool = True):
    """Добавляет абзац с формулой в документ python-docx."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import parse_xml
    from docx.shared import Cm

    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    if align_center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p._p.append(parse_xml(omath_xml(parts, "center" if align_center else "left")))
    return p


def to_text(parts) -> str:
    """Та же формула простым текстом — для модели и для консоли."""
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, Sym):
            out.append(p.base + (f"_{p.sub}" if p.sub else ""))
        elif isinstance(p, Frac):
            out.append(f"({to_text(p.num)})/({to_text(p.den)})")
        elif isinstance(p, System):
            out.append("{ " + "; ".join(to_text(r) for r in p.rows) + " }")
    return "".join(out)
