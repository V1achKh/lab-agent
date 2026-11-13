"""Отрисовка принципиальной схемы по описанию цепи (ГОСТ 2.702, 2.728, 2.729).

Модель сама ничего не рисует: она только описывает цепь (JSON), а картинку
строит этот код. Поэтому рисунок всегда совпадает с тем, что было посчитано.

Раскладка «лестницей», как в большинстве лабораторных схем:
  * узел "0" — нижняя шина;
  * элементы между узлом и землёй стоят вертикально («ступеньки»);
  * элементы между двумя узлами — горизонтально на верхней шине;
  * параллельные элементы на одной паре узлов (например, вольтметр на резисторе)
    поднимаются «мостиком» над шиной.

Условные графические обозначения (размеры в мм, как в ГОСТ):
  резистор — прямоугольник 10×4; конденсатор — две пластины 8 мм с зазором 2 мм;
  катушка — 4 полуокружности; источник ЭДС — окружность ⌀10 со стрелкой к «+»;
  источник тока — окружность ⌀10 с двойной стрелкой; амперметр и вольтметр —
  окружность ⌀10 с буквой A или V.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Circle, Polygon, Rectangle  # noqa: E402

from . import config  # noqa: E402
from .circuit import GROUND, Circuit, Element, element_value_label  # noqa: E402


def _sub(name: str) -> str:
    """R1 → R$_{1}$ (индекс как в заданиях), RV → R$_{V}$, Eдоп → E$_{доп}$."""
    import re

    if name in ("RV", "RA"):
        return f"R$_{{{name[1]}}}$"
    m = re.match(r"^(PV|PA|[A-Za-zА-Яа-я])(.*)$", name)
    if not m or not m.group(2):
        return name
    return f"{m.group(1)}$_{{{m.group(2)}}}$"


def _bmstu_label(e: Element, value: str, vertical: bool = False) -> str:
    """«R₁ = 1 кОм» над горизонтальным элементом; у вертикальных — в две-три строки,
    чтобы подпись не наезжала на соседнюю ветвь."""
    sep = "\n" if vertical else " = "
    if e.type in ("AM", "VM"):
        if e.r:
            rv = f"R$_{{{'V' if e.type == 'VM' else 'A'}}}$"
            return f"{e.name}\n{rv} =\n{value}" if vertical else f"{e.name}, {rv} = {value}"
        return e.name
    if not value:
        return _sub(e.name)
    return f"{_sub(e.name)}{sep}{value}"

SLOT = 34.0   # расстояние между узлами по горизонтали под один элемент
DX = 36.0     # шаг между вертикальными ветвями одного узла (с местом под подпись)
H = 44.0      # высота верхней шины над нижней
LIFT = 20.0   # высота «мостика» для параллельных элементов
LINE = 1.1    # толщина линий, pt
_TEXT_K = 1.0  # увеличение шрифтов для широких схем (выставляется в draw_schematic)
FONT = 9

_BODY = {"R": 10.0, "C": 2.0, "L": 12.0, "V": 10.0, "I": 10.0, "AM": 10.0, "VM": 10.0}


@dataclass
class _Canvas:
    segments: list = field(default_factory=list)   # провода: ((x1,y1),(x2,y2))
    shapes: list = field(default_factory=list)     # функции рисования корпусов
    texts: list = field(default_factory=list)      # (x, y, text, ha, va, size)
    pts: list = field(default_factory=list)        # все точки — для границ рисунка
    node_pos: dict = field(default_factory=dict)   # узел → точка на рисунке (для номеров точек)
    readings: dict = field(default_factory=dict)   # прибор → показание (схема измерения)

    def wire(self, a, b):
        if a != b:
            self.segments.append((a, b))
            self.pts += [a, b]


def _arrowhead(ax, tip, direction, size=2.2, width=1.1):
    dx, dy = direction
    nx, ny = -dy, dx
    bx, by = tip[0] - dx * size, tip[1] - dy * size
    poly = Polygon([tip, (bx + nx * width, by + ny * width), (bx - nx * width, by - ny * width)],
                   closed=True, fc="black", ec="black", lw=0.5)
    ax.add_patch(poly)


def _element(cv: _Canvas, e: Element, p1, p2, label_side: str):
    """Элемент между точками p1 (вывод n1) и p2 (вывод n2). Выводы — провода до корпуса."""
    (x1, y1), (x2, y2) = p1, p2
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if length < 1e-9:      # оба вывода в одной точке (элемент замкнут сам на себя) — рисовать нечего
        return
    d = ((x2 - x1) / length, (y2 - y1) / length)
    n = (-d[1], d[0])
    c = ((x1 + x2) / 2, (y1 + y2) / 2)
    half = _BODY[e.type] / 2

    def P(u, w):
        return (c[0] + u * d[0] + w * n[0], c[1] + u * d[1] + w * n[1])

    a_end, b_start = P(-half, 0), P(half, 0)
    cv.wire(p1, a_end)
    cv.wire(b_start, p2)
    horizontal = abs(d[1]) < 1e-9

    def draw(ax, e=e, P=P, d=d, n=n, c=c, half=half, horizontal=horizontal):
        kw = dict(color="black", lw=LINE, solid_capstyle="butt")
        if e.type == "R":
            w, h = (10, 4) if horizontal else (4, 10)
            ax.add_patch(Rectangle((c[0] - w / 2, c[1] - h / 2), w, h, fill=False, ec="black", lw=LINE))
        elif e.type == "C":
            for u in (-1.0, 1.0):
                a, b = P(u, -4), P(u, 4)
                ax.plot([a[0], b[0]], [a[1], b[1]], **kw)
        elif e.type == "L":
            r = 1.5
            for k in range(4):
                cc = P(-half + r + 2 * r * k, 0)
                ang = 0 if horizontal else 90
                # полуокружности выпуклостью в сторону нормали
                ax.add_patch(Arc(cc, 2 * r, 2 * r, angle=ang, theta1=0, theta2=180, ec="black", lw=LINE))
        else:
            ax.add_patch(Circle(c, 5, fill=False, ec="black", lw=LINE))
            if e.type in ("AM", "VM"):
                ax.text(c[0], c[1], "A" if e.type == "AM" else "V", ha="center", va="center",
                        fontsize=(FONT + 2) * _TEXT_K, family="DejaVu Sans")
            elif e.type == "V":
                # стрелка внутри ЭДС направлена к плюсу (вывод n1)
                tail, tip = P(3.6, 0), P(-3.6, 0)
                ax.plot([tail[0], tip[0]], [tail[1], tip[1]], **kw)
                _arrowhead(ax, tip, (-d[0], -d[1]))
            elif e.type == "I":
                # ток источника тока течёт от n1 к n2: две «галочки» по направлению тока
                for u0 in (-0.6, 1.6):
                    pts = [P(u0 - 1.8, -2.4), P(u0, 0), P(u0 - 1.8, 2.4)]
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], **kw)

    cv.shapes.append(draw)

    # подписи: обозначение и номинал
    value = element_value_label(e)
    off = 6.5 if e.type in ("V", "I", "AM", "VM") else 4.6
    if config.SCHEMATIC_LABELS == "bmstu":
        label = _bmstu_label(e, value, vertical=not horizontal)
        reading = e.name in cv.readings   # схема измерения: показание прибора отдельной строкой
        if reading:
            label = f"показание {cv.readings[e.name]}\n{label}" if horizontal else \
                f"{label}\nпоказание\n{cv.readings[e.name]}"
        if horizontal:
            cv.texts.append((c[0], c[1] + off, label, "center", "bottom", FONT))
            cv.pts += [P(-half, -6), P(half, 6), (c[0], c[1] + off + (16 if reading else 10))]
        else:
            sx = 1 if label_side == "right" else -1
            cv.texts.append((c[0] + sx * off, c[1], label, "left" if sx > 0 else "right", "center", FONT))
            cv.pts += [P(-half, -6), P(half, 6), (c[0] + sx * (off + 20), c[1])]
        return
    if horizontal:
        cv.texts.append((c[0], c[1] + off, e.name, "center", "bottom", FONT))
        if value:
            cv.texts.append((c[0], c[1] - off, value, "center", "top", FONT - 1))
    else:
        sx = 1 if label_side == "right" else -1
        ha = "left" if sx > 0 else "right"
        cv.texts.append((c[0] + sx * off, c[1] + 1.2, e.name, ha, "bottom", FONT))
        if value:
            cv.texts.append((c[0] + sx * off, c[1] - 1.2, value, ha, "top", FONT - 1))
    cv.pts += [P(-half, -6), P(half, 6), (c[0] + 22, c[1]), (c[0] - 4, c[1])]


def _layout(circuit: Circuit, cv: _Canvas):
    els = circuit.elements
    shunts: dict[str, list[Element]] = {}
    series: list[Element] = []
    for e in els:
        if GROUND in (e.n1, e.n2):
            node = e.n2 if e.n1 == GROUND else e.n1
            shunts.setdefault(node, []).append(e)
        else:
            series.append(e)

    # порядок узлов слева направо: обход в глубину по «горизонтальным» элементам от источника
    adj: dict[str, list[str]] = {}
    for e in series:
        adj.setdefault(e.n1, []).append(e.n2)
        adj.setdefault(e.n2, []).append(e.n1)
    src = next((e for e in els if e.is_source), els[0])
    root = src.n1 if src.n1 != GROUND else src.n2
    order: list[str] = []

    def dfs(v):
        order.append(v)
        for w in adj.get(v, []):
            if w not in order:
                dfs(w)

    dfs(root)
    for v in circuit.nodes():
        if v not in order:
            dfs(v)
    col = {v: i for i, v in enumerate(order)}

    # на верхней шине — по одному элементу между соседними узлами, остальное — «мостики»
    rail: dict[tuple, Element] = {}
    bridges: list[Element] = []
    for e in sorted(series, key=lambda e: e.is_meter):
        a, b = sorted((e.n1, e.n2), key=col.get)
        if col[b] - col[a] == 1 and (a, b) not in rail:
            rail[(a, b)] = e
        else:
            bridges.append(e)

    # координаты узлов
    def shunt_order(e: Element):
        return (0 if e.is_source else 2 if e.type == "VM" else 1)

    x_start, x_end, shunt_x = {}, {}, {}
    cursor = 0.0
    for i, v in enumerate(order):
        if i > 0:
            cursor += SLOT
        sh = sorted(shunts.get(v, []), key=shunt_order)
        shunts[v] = sh
        x_start[v] = cursor
        xs = [cursor + k * DX for k in range(len(sh))]
        shunt_x[v] = xs
        x_end[v] = xs[-1] if xs else cursor
        cursor = x_end[v]

    # верхняя шина внутри узла
    for v in order:
        cv.wire((x_start[v], H), (x_end[v], H))
        cv.pts.append((x_start[v], H))
        cv.node_pos[v] = ((x_start[v] + x_end[v]) / 2, H)
    # элементы на шине
    for (a, b), e in rail.items():
        pa, pb = (x_end[a], H), (x_start[b], H)
        p1, p2 = (pa, pb) if e.n1 == a else (pb, pa)
        _element(cv, e, p1, p2, "right")
    # вертикальные ветви
    all_x = []
    for v in order:
        for e, x in zip(shunts[v], shunt_x[v]):
            top, bot = (x, H), (x, 0.0)
            p1, p2 = (top, bot) if e.n1 == v else (bot, top)
            side = "left" if (e.is_source and x == min(min(xs) for xs in shunt_x.values() if xs)) else "right"
            _element(cv, e, p1, p2, side)
            all_x.append(x)
    if all_x:
        cv.wire((min(all_x), 0.0), (max(all_x), 0.0))
        cv.node_pos[GROUND] = ((min(all_x) + max(all_x)) / 2, 0.0)
    # «мостики» над шиной: короткие ниже, длинные выше
    levels: list[list[tuple]] = []
    for e in sorted(bridges, key=lambda e: abs(col[e.n1] - col[e.n2])):
        a, b = sorted((e.n1, e.n2), key=col.get)
        xa, xb = x_end[a], x_start[b]
        lvl = 0
        while lvl < len(levels) and any(not (xb <= s or xa >= f) for s, f in levels[lvl]):
            lvl += 1
        if lvl == len(levels):
            levels.append([])
        levels[lvl].append((xa, xb))
        y = H + LIFT * (lvl + 1)
        cv.wire((xa, H), (xa, y))
        cv.wire((xb, H), (xb, y))
        p1, p2 = ((xa, y), (xb, y)) if e.n1 == a else ((xb, y), (xa, y))
        _element(cv, e, p1, p2, "right")


def _junctions(segments) -> list[tuple]:
    """Точка соединения ставится там, где сходятся три и более проводов."""
    cand = set()
    for a, b in segments:
        cand.add(a)
        cand.add(b)
    dots = []
    eps = 1e-6
    for p in cand:
        k = 0
        for a, b in segments:
            if abs(a[0] - p[0]) < eps and abs(a[1] - p[1]) < eps or abs(b[0] - p[0]) < eps and abs(b[1] - p[1]) < eps:
                k += 1
            elif _inside(p, a, b):
                k += 2
        if k >= 3:
            dots.append(p)
    return dots


def _inside(p, a, b) -> bool:
    eps = 1e-6
    if abs(a[0] - b[0]) < eps:  # вертикальный отрезок
        return abs(p[0] - a[0]) < eps and min(a[1], b[1]) + eps < p[1] < max(a[1], b[1]) - eps
    if abs(a[1] - b[1]) < eps:  # горизонтальный
        return abs(p[1] - a[1]) < eps and min(a[0], b[0]) + eps < p[0] < max(a[0], b[0]) - eps
    return False


def draw_schematic(circuit: Circuit, out_png: Path, out_svg: Path | None = None,
                   node_labels: dict | None = None, readings: dict | None = None) -> Path:
    """node_labels — {узел: номер точки}: номера точек соединения (для уравнений через потенциалы);
    readings — {прибор: показание}: подпись показания (схема измерения)."""
    cv = _Canvas(readings=dict(readings or {}))
    _layout(circuit, cv)
    if node_labels:
        for n, (x, y) in cv.node_pos.items():
            if n in node_labels:
                cv.pts.append((x, y + 7 if y > 0 else y - 7))
    xs = [p[0] for p in cv.pts]
    ys = [p[1] for p in cv.pts]
    x0, x1 = min(xs) - 14, max(xs) + 6
    y0, y1 = min(ys) - 8, max(ys) + 8
    scale = 1.25 / 25.4  # мм → дюймы, с небольшим увеличением для читаемости
    # широкую схему в отчёте уменьшат до ширины страницы (~16 см) — заранее увеличим шрифт и линии,
    # чтобы после уменьшения подписи были ~8 пт
    k = max(1.0, (x1 - x0) * 1.25 / 160)
    fig = plt.figure(figsize=((x1 - x0) * scale, (y1 - y0) * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    global LINE, _TEXT_K
    base_line, LINE = LINE, LINE * k
    _TEXT_K = min(k, 1.6)
    for a, b in cv.segments:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=LINE, solid_capstyle="round")
    for draw in cv.shapes:
        draw(ax)
    LINE, _TEXT_K = base_line, 1.0
    for p in _junctions(cv.segments):
        ax.add_patch(Circle(p, 0.9, color="black"))
    for n, (x, y) in (cv.node_pos.items() if node_labels else []):
        if n not in node_labels:
            continue
        ax.add_patch(Circle((x, y), 0.9, color="black"))
        up = y > 0
        ax.text(x, y + (1.6 if up else -1.6), str(node_labels[n]), ha="center", va="bottom" if up else "top",
                fontsize=FONT * min(k, 1.6), family="DejaVu Sans", weight="bold")
    for x, y, t, ha, va, size in cv.texts:
        ax.text(x, y, t, ha=ha, va=va, fontsize=min(size * k, size * 1.6), family="DejaVu Sans", style="italic")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor="white")
    if out_svg:
        fig.savefig(out_svg, facecolor="white")
    plt.close(fig)
    return out_png
