"""Лабораторная работа «Цепи постоянного тока» (расчёт и погрешность измерения).

Делает всё, что требует методичка, детерминированно, без языковой модели:
  1. система уравнений по правилам Кирхгофа — генерируется из схемы (формулы Word);
  2. решение системы и искомое напряжение U0 (или ток I0) при идеальном приборе;
  3. «измерение» прибором с заданным внутренним сопротивлением, показание с 4 значащими цифрами;
  4. погрешность Δ = |U − U0| (не меньше инструментальной — половины последнего разряда);
  5. сопротивление схемы r относительно клемм прибора, δ = r / Rv (или δ = Ra / r), Δ0 = δ·U;
  6. сравнение Δ и Δ0, при необходимости — уточняющее измерение с добавочным источником.
"""
from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field

from . import config
from .circuit import GROUND, Circuit, Element, fmt_si
from .omml import Frac, Sym, System, to_text
from .solver import operating_point, thevenin_resistance

DIGITS = int(getattr(config, "METER_DIGITS", 4))


# ---------------------------------------------------------------- числа

def round_sig(x: float, digits: int = DIGITS) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


def half_lsd(x: float, digits: int = DIGITS) -> float:
    """Инструментальная погрешность: половина последнего разряда показания."""
    if x == 0:
        return 0.0
    return 0.5 * 10 ** (int(math.floor(math.log10(abs(x)))) - digits + 1)


def num(x: float, digits: int = 6) -> str:
    """Число по-русски: запятая, без лишних нулей, без «e-05»."""
    if x == 0:
        return "0"
    if not math.isfinite(x):
        return "∞"
    ax = abs(x)
    if 1e-4 <= ax < 1e7:
        dec = max(0, digits - 1 - int(math.floor(math.log10(ax))))
        s = f"{x:.{dec}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return config.dsep(s).replace("-", "−")
    m, e = f"{x:.{digits - 1}e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{config.dsep(m)}·10^{int(e)}".replace("-", "−")


def sym(name: str, base: str | None = None) -> Sym:
    """R1 → R с индексом 1; J → J; PV1 → PV с индексом 1. base — заменить букву (I для тока)."""
    m = re.match(r"^([A-Za-zА-Яа-я]+)(.*)$", name)
    letters, rest = (m.group(1), m.group(2)) if m else (name, "")
    if base:
        return Sym(base, name)
    return Sym(letters, rest)


# ---------------------------------------------------------------- граф ветвей

@dataclass
class Branch:
    e: Element
    a: str  # положительное направление тока: a → b через элемент
    b: str

    @property
    def name(self):
        return self.e.name


@dataclass
class Kirchhoff:
    kcl: list = field(default_factory=list)         # формулы первого правила (символьно)
    kvl: list = field(default_factory=list)         # формулы второго правила
    kcl_num: list = field(default_factory=list)     # с подставленными значениями
    kvl_num: list = field(default_factory=list)
    kcl_nodes: list = field(default_factory=list)
    loops: list = field(default_factory=list)       # описание контуров словами
    unknowns: list = field(default_factory=list)    # имена ветвей с неизвестным током
    solution: dict = field(default_factory=dict)    # ток каждой ветви


def _branches(circuit: Circuit) -> list[Branch]:
    out = []
    for e in circuit.elements:
        if e.type == "VM":
            continue  # идеальный вольтметр — разрыв
        if e.type == "V":
            out.append(Branch(e, e.n2, e.n1))  # внутри источника ЭДС ток от «−» к «+»
        else:
            out.append(Branch(e, e.n1, e.n2))
    return out


def _spanning_forest(nodes: list[str], branches: list[Branch]):
    """Дерево графа (без источников тока). parent[node] = (ветвь, узел-родитель)."""
    adj: dict[str, list[tuple[Branch, str]]] = {n: [] for n in nodes}
    for br in branches:
        adj[br.a].append((br, br.b))
        adj[br.b].append((br, br.a))
    parent, depth, tree = {}, {}, set()
    for root in nodes:
        if root in depth:
            continue
        depth[root] = 0
        parent[root] = None
        q = deque([root])
        while q:
            v = q.popleft()
            for br, w in sorted(adj[v], key=lambda t: (t[0].e.type != "V", t[0].name)):
                if w not in depth:
                    depth[w] = depth[v] + 1
                    parent[w] = (br, v)
                    tree.add(id(br))
                    q.append(w)
    return parent, depth, tree


def _tree_path(u: str, v: str, parent, depth) -> list[tuple[Branch, int]] | None:
    """Путь u → v по дереву: [(ветвь, знак)], знак +1 — идём по направлению ветви a → b."""
    up_u, up_v = [], []
    x, y = u, v
    while x != y:
        if depth.get(x) is None or depth.get(y) is None:
            return None
        if depth[x] >= depth[y]:
            if parent[x] is None:
                return None
            br, p = parent[x]
            up_u.append((br, x, p))
            x = p
        else:
            if parent[y] is None:
                return None
            br, p = parent[y]
            up_v.append((br, p, y))
            y = p
    path = []
    for br, frm, to in up_u + list(reversed(up_v)):
        path.append((br, 1 if (br.a == frm and br.b == to) else -1))
    return path


def _lin(terms: list[tuple[int, list]]) -> list:
    """[(знак, [кусочки])] → «a − b + c». Пусто → «0»."""
    out = []
    for k, (s, parts) in enumerate(terms):
        if k == 0:
            out += (["−"] if s < 0 else []) + parts
        else:
            out += [" − " if s < 0 else " + "] + parts
    return out or ["0"]


def kirchhoff(circuit: Circuit, ideal: Circuit) -> Kirchhoff:
    """Система уравнений по правилам Кирхгофа для цепи с идеальными приборами."""
    branches = _branches(ideal)
    nodes = []
    for br in branches:
        for n in (br.a, br.b):
            if n not in nodes:
                nodes.append(n)
    nodes.sort(key=lambda n: (n != GROUND, n))
    no_j = [br for br in branches if br.e.type != "I"]
    parent, depth, tree = _spanning_forest(nodes, no_j)
    k = Kirchhoff()
    k.unknowns = [br.name for br in branches if br.e.type != "I"]

    def cur(br: Branch):
        return sym(br.name) if br.e.type == "I" else sym(br.name, "I")

    # первое правило: для всех узлов, кроме опорного (сумма вытекающих токов = 0)
    for n in nodes:
        if n == GROUND:
            continue
        terms, terms_num, rhs = [], [], 0.0
        for br in branches:
            s = 1 if br.a == n else -1 if br.b == n else 0
            if s == 0:
                continue
            terms.append((s, [cur(br)]))
            if br.e.type == "I":
                rhs -= s * br.e.value
            else:
                terms_num.append((s, [cur(br)]))
        if not terms:
            continue
        k.kcl_nodes.append(n)
        k.kcl.append(_lin(terms) + [" = 0"])
        k.kcl_num.append(_lin(terms_num) + [f" = {num(rhs)}"])

    # второе правило: по одному контуру на каждую ветвь, не вошедшую в дерево
    for br in no_j:
        if id(br) in tree:
            continue
        path = _tree_path(br.b, br.a, parent, depth)
        if path is None:
            continue
        loop = [(br, 1)] + path
        lhs, lhs_num, rhs, rhs_num, rhs_val = [], [], [], [], 0.0
        for b2, s in loop:
            if b2.e.type == "R":
                lhs.append((s, [sym(b2.name), cur(b2)]))
                lhs_num.append((s, [f"{num(b2.e.value)}·", cur(b2)]))
            elif b2.e.type == "V":
                rhs.append((s, [sym(b2.name)]))
                rhs_val += s * (b2.e.value or 0.0)
        if not lhs and not rhs:
            continue
        k.kvl.append(_lin(lhs) + [" = "] + _lin(rhs))
        k.kvl_num.append(_lin(lhs_num) + [f" = {num(rhs_val)}"])
        k.loops.append(" → ".join(b2.name for b2, _ in loop))

    # решение (численно, тем же решателем) — токи в выбранных направлениях
    op = operating_point(ideal)
    for br in branches:
        val = op["currents"][br.name]
        if br.e.type == "V":
            val = -val  # в решателе ток источника ЭДС считается от «+» к «−»
        k.solution[br.name] = val
    k.potentials = op["nodes"]
    k.tree_info = (parent, depth)
    return k


# ---------------------------------------------------------------- лабораторная целиком

@dataclass
class DCResult:
    circuit: Circuit
    meter: Element
    quantity: str            # "U" или "I"
    unit: str                # "В" или "А"
    k: Kirchhoff
    target_formula: list     # U0 = ... по пути в дереве
    target_formula_num: list
    x0: float                # расчётное значение (идеальный прибор)
    x_full: float            # показание реального прибора, полная точность
    x_disp: float            # показание прибора (4 значащие цифры)
    delta: float             # |x − x0| с учётом инструментальной
    delta_raw: float
    instr: float             # инструментальная погрешность
    r: float                 # сопротивление относительно клемм прибора
    rel: float               # δ
    delta0: float            # Δ0
    ok_first: bool
    refine: dict | None      # уточняющее измерение
    target_text: str


def is_dc_lab(circuit: Circuit) -> bool:
    return (len(circuit.meters()) == 1 and all(a.type == "op" for a in circuit.analyses)
            and all(e.type in ("R", "V", "I", "AM", "VM") for e in circuit.elements))


def _what(circuit: Circuit, m: Element) -> str:
    """«падение напряжения на источнике E1» / «напряжение между узлами 2 и 0» / «ток через R3»."""
    if m.type == "VM":
        same = [e for e in circuit.elements if e is not m and {e.n1, e.n2} == {m.n1, m.n2}]
        if same:
            e = same[0]
            kind = {"R": "резисторе", "V": "источнике напряжения", "I": "источнике тока"}.get(e.type, "элементе")
            return f"падение напряжения на {kind} {e.name}"
        return f"напряжение между узлами {m.n1} и {m.n2}"
    # амперметр: ток через соседний последовательный элемент
    neigh = [e for e in circuit.elements if e is not m and e.type != "VM" and (m.n2 in (e.n1, e.n2))
             and sum(1 for x in circuit.elements if x.type != "VM" and m.n2 in (x.n1, x.n2)) == 2]
    if neigh:
        return f"ток через {'резистор' if neigh[0].type == 'R' else 'элемент'} {neigh[0].name}"
    return f"ток в ветви между узлами {m.n1} и {m.n2}"


# ---------------------------------------------------------------- прибор там, где требует задание

_LAT = str.maketrans("АВЕКМНОРСТХ", "ABEKMHOPCTX")
_V_RE = re.compile(r"(?:падени\w*\s+напряжени\w*|напряжени\w*)\s+на\s+(?:зажимах\s+)?"
                   r"(?:резистор\w*|источник\w*(?:\s+(?:напряжения|тока|ЭДС))?|элемент\w*|ЭДС)?\s*"
                   r"([A-ZА-Я]{1,3}\d+)")
_I_RE = re.compile(r"\bток\w*\s*,?\s*(?:протекающ\w+\s+)?(?:через|в\s+ветви\s+(?:с\s+)?)\s*"
                   r"(?:резистор\w*|элемент\w*|источник\w*(?:\s+(?:напряжения|тока|ЭДС))?)?\s*"
                   r"([A-ZА-Я]{1,3}\d+)")


def task_target(task: str) -> tuple[str, str] | None:
    """Что нужно определить по тексту задания: ("V", "E1") — напряжение на E1, ("I", "R3") — ток через R3.
    Ищется только в самом задании, не в приложенной методичке."""
    head = task.split("Методические указания к этой работе")[0]
    found = []
    for kind, rx in (("V", _V_RE), ("I", _I_RE)):
        m = rx.search(head)
        if m:
            found.append((m.start(), kind, m.group(1).translate(_LAT).upper()))
    if not found:
        return None
    _, kind, name = min(found)
    return kind, name


def structure_problems(circuit: Circuit) -> str | None:
    """Ошибки чтения схемы лабы 1, которые видны без эталона. Текст — для модели (самоисправление)."""
    if not is_dc_lab(circuit):
        return None
    m = circuit.meters()[0]
    probs = []
    if not m.r:
        probs.append(f"у прибора {m.name} не указано внутреннее сопротивление r (или оно 0). В лабораторной №1 оно "
                     "всегда задано: число рядом с прибором на картинке (например «0,05» → r = 0.05 Ом, "
                     "«500 к» → r = 500000) или в тексте задания")
    if m.type == "AM":
        shorted = [e.name for e in circuit.elements if e is not m and e.type != "VM" and {e.n1, e.n2} == {m.n1, m.n2}]
        if shorted:
            probs.append(f"амперметр {m.name} подключён к тем же двум точкам, что и {', '.join(shorted)}, — то есть "
                         f"параллельно, и закорачивает {'его' if len(shorted) == 1 else 'их'}. Амперметр включается "
                         "ПОСЛЕДОВАТЕЛЬНО, в разрыв провода: заведи новую точку между амперметром и элементом, "
                         "через который идёт измеряемый ток")
    if m.type == "AM" and not probs:
        try:
            r = thevenin_resistance(circuit, m.n1, m.n2, remove=(m.name,))
        except Exception:
            r = 0.0
        if math.isinf(r) or r > 1e12:
            probs.append(f"амперметр {m.name} оказался в одной ветви с источником тока: ток через него всегда "
                         "равен току источника, измерять нечего — значит, схема прочитана неверно. Ещё раз "
                         "пройди по проводам на картинке: что соединено с каждым выводом амперметра")
    if m.type == "VM":
        series = [x for x in (m.n1, m.n2) if x != GROUND and
                  sum(1 for e in circuit.elements if e is not m and x in (e.n1, e.n2)) < 1]
        if series:
            probs.append(f"вольтметр {m.name} подключён к точке {series[0]}, к которой больше ничего не подключено. "
                         "Вольтметр подключается ПАРАЛЛЕЛЬНО участку — к двум точкам, где уже есть элементы")
    return "; ".join(probs) or None


def target_mismatch(task: str, circuit: Circuit) -> str | None:
    """Проверка кодом, что прибор подключён туда, куда требует задание. Текст ошибки — для модели."""
    t = task_target(task or "")
    meters = circuit.meters()
    if t is None or len(meters) != 1:
        return None
    kind, name = t
    m = meters[0]
    el = next((e for e in circuit.elements if e.name.translate(_LAT).upper() == name), None)
    if el is None:
        return f"по заданию нужно измерить величину на элементе {name}, а в цепи такого элемента нет"
    if kind == "V":
        if m.type != "VM":
            return f"по заданию нужно определить напряжение на {name} — нужен вольтметр (type VM), а не амперметр"
        if {m.n1, m.n2} == {el.n1, el.n2}:
            return None
        return (f"по заданию нужно определить падение напряжения на {name}, а вольтметр {m.name} подключён "
                f"к узлам {m.n1} и {m.n2}. Вольтметр подключается параллельно {name}: "
                f"n1 = \"{el.n1}\", n2 = \"{el.n2}\" (те же узлы, что у {name}). Остальную цепь не меняй.")
    if m.type != "AM":
        return f"по заданию нужно определить ток через {name} — нужен амперметр (type AM), а не вольтметр"
    for x in (m.n1, m.n2):
        if x != GROUND and x in (el.n1, el.n2) and \
                sum(1 for e in circuit.elements if e.type != "VM" and x in (e.n1, e.n2)) == 2:
            return None
    return (f"по заданию нужно определить ток через {name}, а амперметр {m.name} включён не последовательно "
            f"с {name}. Амперметр включается в разрыв ветви: заведи новый узел между {name} и амперметром, "
            f"например {name}: n1 = \"{el.n1}\", n2 = \"N\"; {m.name}: n1 = \"N\", n2 = \"{el.n2}\". "
            "Остальную цепь не меняй.")


def analyze(circuit: Circuit) -> DCResult:
    meter = circuit.meters()[0]
    is_v = meter.type == "VM"
    ideal = circuit.idealized()
    k = kirchhoff(circuit, ideal)
    real = operating_point(circuit)
    quantity, unit = ("U", "В") if is_v else ("I", "А")

    # расчётное значение и формула для него
    if is_v:
        parent, depth = k.tree_info
        x0 = (k.potentials.get(meter.n1, 0.0) if meter.n1 != GROUND else 0.0) - \
             (k.potentials.get(meter.n2, 0.0) if meter.n2 != GROUND else 0.0)
        path = _tree_path(meter.n1, meter.n2, parent, depth)
        if path:
            terms, terms_num, total = [], [], 0.0
            for br, s in path:
                if br.e.type == "R":
                    terms.append((s, [sym(br.name), sym(br.name, "I")]))
                    terms_num.append((s, [f"{num(br.e.value)}·({num(k.solution[br.name])})"]))
                elif br.e.type == "V":
                    # по направлению a→b внутри источника ЭДС потенциал растёт на E: U_ab = −E
                    terms.append((-s, [sym(br.name)]))
                    terms_num.append((-s, [num(br.e.value)]))
            formula = [Sym("U", "0"), " = "] + _lin(terms)
            # числа подставляем, только если формула не сводится к одному значению ЭДС
            if len(terms_num) > 1 or any("·" in str(x) for _, t in terms_num for x in t):
                formula += [" = "] + _lin(terms_num)
            formula += [f" = {q(x0, 'В')}"]
            formula_num = formula
        else:
            formula = [Sym("U", "0"), " = ", Sym("φ", meter.n1), " − ", Sym("φ", meter.n2), f" = {q(x0, 'В')}"]
            formula_num = formula
    else:
        x0 = k.solution[meter.name]
        formula = [Sym("I", "0"), " = ", sym(meter.name, "I"), f" = {q(x0, 'А')}"]
        formula_num = formula

    x_full = real["readings"][meter.name]
    x_disp = round_sig(x_full)
    instr = half_lsd(x_disp)
    delta_raw = abs(x_disp - x0)
    delta = max(delta_raw, instr)

    r = thevenin_resistance(circuit, meter.n1, meter.n2, remove=(meter.name,))
    if is_v:
        rel = 0.0 if r == 0 else (r / meter.r if meter.r else 0.0)
    else:
        rel = 0.0 if not meter.r else (0.0 if math.isinf(r) else (meter.r / r if r > 0 else math.inf))
    delta0 = rel * abs(x_disp)
    tol = 1e-9 * max(abs(x0), 1e-12)
    ok_first = delta <= delta0 + tol

    refine = None
    if not ok_first:
        # уточнение по методичке: добавочный источник с напряжением (током) = первому показанию
        x1 = x_disp
        rc = _refine_circuit(circuit, meter, x1)
        # показание идеального прибора в схеме уточнения — честная симуляция этой схемы
        x2_full = operating_point(rc)["readings"][meter.name]
        x2 = round_sig(x2_full) if abs(x2_full) > 1e-12 * max(1.0, abs(x1)) else 0.0
        x_ref = x1 + x2
        instr2 = half_lsd(x2) if x2 else 0.0
        d_raw2 = abs(x_ref - x0)
        # По методичке Δ0 = δ·Uизмер. После уточнения измеренный результат — U = U1 + U2, поэтому Δ0
        # пересчитывается по нему. (Δ0 по первому, округлённому показанию чуть меньше истинной
        # методической погрешности — и честное уточнение выглядело бы как «Δ > Δ0» на сотые доли.)
        delta0_2 = rel * abs(x_ref)
        refine = {"x1": x1, "x2": x2, "x": x_ref, "instr": instr2, "delta": max(d_raw2, 0.0),
                  "delta0": delta0_2, "ok": d_raw2 <= delta0_2 + instr2 + tol, "circuit": rc}
    return DCResult(circuit=circuit, meter=meter, quantity=quantity, unit=unit, k=k,
                    target_formula=formula, target_formula_num=formula_num, x0=x0, x_full=x_full,
                    x_disp=x_disp, delta=delta, delta_raw=delta_raw, instr=instr, r=r, rel=rel,
                    delta0=delta0, ok_first=ok_first, refine=refine, target_text=_what(circuit, meter))


def _refine_circuit(circuit: Circuit, meter: Element, x1: float) -> Circuit:
    """Схема уточняющего измерения (как в методичке): к идеальному прибору добавлен
    источник с напряжением U1 (или тока I1), а внутреннее сопротивление прибора
    вынесено отдельным резистором."""
    els = [e for e in circuit.elements if e is not meter]
    m = meter.name
    if meter.type == "VM":
        mid = f"{meter.n1}д"
        els += [Element(name="Eдоп", type="V", n1=meter.n1, n2=mid, value=x1),
                Element(name=m, type="VM", n1=mid, n2=meter.n2),
                Element(name="RV", type="R", n1=meter.n1, n2=meter.n2, value=meter.r)]
    else:
        mid = f"{meter.n1}д"
        els += [Element(name="RA", type="R", n1=meter.n1, n2=mid, value=meter.r or 1e-9),
                Element(name=m, type="AM", n1=mid, n2=meter.n2),
                Element(name="Jдоп", type="I", n1=mid, n2=meter.n2, value=x1)]
    from .circuit import Analysis

    # model_construct: у идеального вольтметра с добавочным источником «висящий» узел — так и задумано
    return Circuit.model_construct(title=circuit.title + " (уточнение)", elements=els,
                                   analyses=[Analysis(type="op")])


# ---------------------------------------------------------------- схема без прибора, точки соединения
#
# Требования преподавателя («Типичные ошибки студентов…», лаб. 1):
#   • измерительный прибор не является элементом схемы — он лишь показывает, что и чем измерить;
#   • в описании схемы нельзя упоминать узлы, «слева/справа/верх/низ», номера выводов;
#     описание заканчивается словами «Требуется определить …»;
#   • метод правил Кирхгофа: правило узлов для каждой точки и закон Ома для каждого резистора
#     (токи и потенциалы).

def physical(circuit: Circuit) -> tuple[Circuit, dict]:
    """Схема без измерительного прибора: вольтметр убирается, амперметр заменяется проводом.
    Возвращает (схема, {узел исходной цепи → узел схемы})."""
    m = circuit.meters()[0]
    merge: dict[str, str] = {}
    if m.type == "AM":
        keep, drop = (m.n2, m.n1) if m.n1 != GROUND and m.n2 == GROUND else (m.n1, m.n2)
        if drop == GROUND:
            keep, drop = drop, keep
        merge[drop] = keep
    els = [e.model_copy(update={"n1": merge.get(e.n1, e.n1), "n2": merge.get(e.n2, e.n2)})
           for e in circuit.elements if e is not m]
    els = [e for e in els if e.n1 != e.n2]   # элемент, закороченный проводом вместо амперметра, тока не несёт
    from .circuit import Analysis

    pc = Circuit.model_construct(title=circuit.title, elements=els, analyses=[Analysis(type="op")])
    return pc, merge


def point_names(pc: Circuit) -> dict[str, str]:
    """Точки соединения схемы → номера 1, 2, … (общая точка — 0). Номера отмечаются на рисунке."""
    names = {GROUND: "0"}
    for k, n in enumerate(pc.nodes(), 1):
        names[n] = str(k)
    return names


_KIND = {"R": "резистор", "V": "источник напряжения", "I": "источник тока"}


class _Terms:
    """Словесные описания выводов элементов (без номеров выводов)."""

    def __init__(self):
        self.mentioned: set[str] = set()

    def value(self, e: Element) -> str:
        if e.type == "R":
            return fmt_si(e.value, "Ом")
        return fmt_si(e.value or 0, "В" if e.type == "V" else "А")

    def term(self, e: Element, node: str, case: str) -> tuple[str, str]:
        """(текст, род: m|f) — вывод элемента e в точке node; case: nom | ins | dat | acc."""
        first = e.name not in self.mentioned
        self.mentioned.add(e.name)
        val = f" ({self.value(e)})" if first else ""
        if e.type == "V":
            plus = node == e.n1
            adj = {"nom": ("положительная", "отрицательная"), "ins": ("положительной", "отрицательной"),
                   "dat": ("положительной", "отрицательной"), "acc": ("положительную", "отрицательную")}[case]
            noun = {"nom": "клемма", "ins": "клеммой", "dat": "клемме", "acc": "клемму"}[case]
            src = "источника напряжения" if first else "источника"
            return f"{adj[0 if plus else 1]} {noun} {src} {e.name}{val}", "f"
        if e.type == "I":
            out = node == e.n2   # ток источника течёт от n1 к n2 — в точку n2 он вытекает
            noun = {"nom": "клемма", "ins": "клеммой", "dat": "клемме", "acc": "клемму"}[case]
            how = "из которой ток вытекает" if out else "в которую ток втекает"
            src = "источника тока" if first else "источника"
            return f"{noun} {src} {e.name}{val}, {how}", "f"
        noun = {"nom": "вывод", "ins": "выводом", "dat": "выводу", "acc": "вывод"}[case]
        if not first:
            adj = {"nom": "свободный ", "ins": "свободным ", "dat": "свободному ", "acc": "свободный "}[case]
            return f"{adj}{noun} резистора {e.name}", "m"
        return f"{noun} резистора {e.name}{val}", "m"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " и " + items[-1]


def point_text(pc: Circuit, node: str, exclude: tuple = ()) -> str:
    """«точке соединения резисторов R3, R4 и источника тока J» — описание точки без слова «узел»."""
    names = [e.name for e in pc.elements if node in (e.n1, e.n2) and e.name not in exclude]
    return "точке соединения элементов " + _join(names) if len(names) > 1 else \
        (f"выводу элемента {names[0]}" if names else "общей точке")


def target_phrase(circuit: Circuit, pc: Circuit, merge: dict) -> str:
    """Конец описания схемы: «требуется определить падение напряжения на … / ток через …»."""
    m = circuit.meters()[0]
    if m.type == "VM":
        same = [e for e in pc.elements if {e.n1, e.n2} == {m.n1, m.n2}]
        if same:
            e = same[0]
            kind = {"R": "резисторе", "V": "источнике напряжения", "I": "источнике тока"}[e.type]
            return f"падение напряжения на {kind} {e.name}"
        return (f"разность потенциалов между {point_text(pc, m.n1).replace('точке', 'точкой')} "
                f"и {point_text(pc, m.n2).replace('точке', 'точкой')}")
    el = series_element(circuit, m)
    if el is not None:
        return f"ток через {_KIND[el.type].split()[0] + ('' if el.type == 'R' else ' ' + ' '.join(_KIND[el.type].split()[1:]))} {el.name}"
    return f"ток в проводе, соединяющем элементы {', '.join(e.name for e in circuit.elements if m.n1 in (e.n1, e.n2) and e is not m)} " \
           f"с элементами {', '.join(e.name for e in circuit.elements if m.n2 in (e.n1, e.n2) and e is not m)}"


def series_element(circuit: Circuit, m: Element) -> Element | None:
    """Элемент, последовательно с которым включён амперметр (единственный другой элемент в точке)."""
    for x in (m.n2, m.n1):
        if x == GROUND:
            continue
        others = [e for e in circuit.elements if e is not m and e.type != "VM" and x in (e.n1, e.n2)]
        if len(others) == 1 and others[0].type in ("R", "V", "I"):
            return others[0]
    return None


def describe_circuit(circuit: Circuit) -> list[str]:
    """«Описание схемы» по требованиям преподавателя: только физические элементы, их клеммы
    и соединения; без узлов, без сторон рисунка, без номеров выводов; прибора в описании нет;
    в конце — что требуется определить."""
    pc, merge = physical(circuit)
    t = _Terms()
    # порядок точек: от положительной клеммы первого источника напряжения, дальше по соседству
    first_src = next((e for e in pc.elements if e.type == "V"), pc.elements[0])
    order, queue = [], [first_src.n1]
    while queue:
        n = queue.pop(0)
        if n in order:
            continue
        order.append(n)
        for e in pc.elements:
            if n in (e.n1, e.n2):
                queue.append(e.n2 if e.n1 == n else e.n1)
    order += [n for n in pc.nodes() + [GROUND] if n not in order]

    sentences = []
    for n in order:
        here = [e for e in pc.elements if n in (e.n1, e.n2)]
        if len(here) < 2:
            continue
        # ведущим в предложении делаем уже упомянутый элемент (так описание читается «по цепочке»)
        here.sort(key=lambda e: (e.name not in t.mentioned, e.type != "V", e.name))
        lead_txt, gender = t.term(here[0], n, "nom")
        rest = [t.term(e, n, "ins")[0] for e in here[1:]]
        verb = "соединена с" if gender == "f" else "соединён с"
        s = f"{lead_txt[0].upper()}{lead_txt[1:]} {verb} {_join(rest)}."
        s = re.sub(r"(ток (?:в|вы)текает)(?= )", r"\1,", s)      # придаточное закрывается запятой
        s = s.replace(" с свободн", " со свободн")
        sentences.append(s)
    sentences.append(f"Требуется определить {target_phrase(circuit, pc, merge)}.")
    return sentences


def measurement_text(res: "DCResult") -> list[str]:
    """«Описание процедуры измерений» (п. 8 требований): как подключается прибор и что он показал."""
    circuit, m = res.circuit, res.meter
    pc, _ = physical(circuit)
    rv = q(m.r or 0, "Ом")
    if m.type == "VM":
        same = [e for e in pc.elements if {e.n1, e.n2} == {m.n1, m.n2}]
        how = (f"подключается параллельно с {('резистором' if same[0].type == 'R' else 'источником напряжения' if same[0].type == 'V' else 'источником тока')} {same[0].name}"
               if same else "подключается к двум точкам схемы")
        lines = [f"Вольтметр постоянного тока с внутренним сопротивлением {rv} {how}. "
                 f"Положительная клемма вольтметра подключается к {point_text(pc, m.n1)}, "
                 f"отрицательная — к {point_text(pc, m.n2)}."]
    else:
        el = series_element(circuit, m)
        side = m.n1 if el is not None and m.n2 in (el.n1, el.n2) else m.n2
        other = [e.name for e in circuit.elements if side in (e.n1, e.n2) and e is not m and e.type != "VM"]
        what = f"последовательно с {'резистором' if el and el.type == 'R' else 'элементом'} {el.name}" if el else "в разрыв провода"
        lines = [f"Амперметр постоянного тока с внутренним сопротивлением {rv} включается {what} — "
                 f"в разрыв провода, соединяющего {el.name if el else 'элементы'} с "
                 f"{'точкой соединения элементов ' + _join(other) if len(other) > 1 else 'элементом ' + other[0] if other else 'общей точкой'}. "
                 f"Положительная клемма амперметра подключается со стороны "
                 f"{'точки соединения элементов ' + _join(other) if len(other) > 1 else 'элемента ' + (other[0] if other else '')}"
                 f"{', отрицательная — к элементу ' + el.name if el else ''}."]
    dev = "вольтметра" if m.type == "VM" else "амперметра"
    lines.append(f"Показания {dev} (с {DIGITS} значащими цифрами, конечные нули сохранены): "
                 f"{res.quantity} = {disp(res.x_disp, res.unit)}.")
    return lines


@dataclass
class NodeEquations:
    """Система по методу правил Кирхгофа в форме, которую требует преподаватель."""
    names: dict                                      # точка схемы → номер
    sources: list = field(default_factory=list)      # φ+ − φ− = E
    kcl: list = field(default_factory=list)          # правило узлов
    ohm: list = field(default_factory=list)          # закон Ома для резисторов
    sources_num: list = field(default_factory=list)
    kcl_num: list = field(default_factory=list)
    ohm_num: list = field(default_factory=list)
    potentials: dict = field(default_factory=dict)   # номер точки → φ
    currents: dict = field(default_factory=dict)     # элемент → ток (R: n1→n2; V: от «−» к «+» внутри)


def node_equations(circuit: Circuit) -> NodeEquations:
    pc, _ = physical(circuit)
    nm = point_names(pc)
    ne = NodeEquations(names=nm)

    def phi(n):
        return Sym("φ", nm[n])

    def cur(e):
        return sym(e.name) if e.type == "I" else sym(e.name, "I")

    op = operating_point(pc)
    pot = {n: (0.0 if n == GROUND else op["nodes"].get(n, 0.0)) for n in [GROUND] + pc.nodes()}
    ne.potentials = {nm[n]: v for n, v in pot.items()}
    for e in pc.elements:
        if e.type == "R":
            ne.currents[e.name] = (pot[e.n1] - pot[e.n2]) / e.value
        elif e.type == "V":
            ne.currents[e.name] = -op["currents"][e.name]   # от «−» к «+» внутри источника
        elif e.type == "I":
            ne.currents[e.name] = e.value

    ne.sources.append([phi(GROUND), " = 0"])
    ne.sources_num.append([phi(GROUND), " = 0"])
    for e in pc.elements:
        if e.type == "V":
            ne.sources.append([phi(e.n1), " − ", phi(e.n2), " = ", sym(e.name)])
            ne.sources_num.append([phi(e.n1), " − ", phi(e.n2), f" = {num(e.value)}"])
    for n in pc.nodes():                       # правило узлов: сумма вытекающих токов равна нулю
        terms, terms_num, rhs = [], [], 0.0
        for e in pc.elements:
            if n not in (e.n1, e.n2):
                continue
            a, b = (e.n2, e.n1) if e.type == "V" else (e.n1, e.n2)
            s = 1 if a == n else -1
            terms.append((s, [cur(e)]))
            if e.type == "I":
                rhs -= s * e.value
            else:
                terms_num.append((s, [cur(e)]))
        ne.kcl.append(_lin(terms) + [" = 0"])
        ne.kcl_num.append(_lin(terms_num) + [f" = {num(rhs)}"])
    for e in pc.elements:                      # закон Ома для каждого резистора
        if e.type != "R":
            continue
        ne.ohm.append([sym(e.name, "I"), " = ", Frac([phi(e.n1), " − ", phi(e.n2)], [sym(e.name)])])
        ne.ohm_num.append([sym(e.name, "I"), " = ", Frac([phi(e.n1), " − ", phi(e.n2)], [num(e.value)])])
    return ne


def text_summary(res: DCResult) -> str:
    """Короткая сводка для языковой модели (из неё модель пишет вывод)."""
    q_, u = res.quantity, res.unit
    lines = [f"Определяется: {res.target_text}.",
             f"Расчёт по правилам Кирхгофа: {q_}0 = {q(res.x0, u)}.",
             f"Прибор {res.meter.name}: внутреннее сопротивление {fmt_si(res.meter.r or 0, 'Ом')}, "
             f"показание {disp(res.x_disp, u)}.",
             f"Погрешность измерения Δ = {q(res.delta, u, 4)}"
             + (" (взята инструментальная погрешность)" if res.delta_raw < res.instr else "") + ".",
             f"Сопротивление схемы относительно клемм прибора r = {q(res.r, 'Ом')}; δ = {num(res.rel)}; "
             f"теоретически возможная погрешность Δ0 = {q(res.delta0, u, 4)}.",
             "Δ ≤ Δ0 — работа выполнена верно." if res.ok_first else "Δ > Δ0 — нужно уточняющее измерение."]
    if res.refine:
        rf = res.refine
        lines.append(f"Уточнение: {q_}1 = {disp(rf['x1'], u)}, {q_}2 = {disp(rf['x2'], u) if rf['x2'] else '0'}, "
                     f"{q_} = {q(rf['x'], u, 10)}, новое Δ = {q(rf['delta'], u, 4)}, "
                     f"Δ0 по уточнённому результату = {q(rf['delta0'], u, 4)} — "
                     + ("Δ не превышает Δ0." if rf["ok"] else "Δ всё ещё больше Δ0."))
    return "\n".join(lines)


def final_phrase(res: DCResult) -> str:
    """Фраза, которой по требованию преподавателя должен заканчиваться отчёт (п. 18)."""
    d = res.refine["delta"] if res.refine else res.delta
    d0 = res.refine["delta0"] if res.refine else res.delta0
    ok = res.refine["ok"] if res.refine else res.ok_first
    if not ok:
        return ("Расхождение теоретических и измеренных значений превосходит теоретически допустимую "
                "погрешность — расчёт или измерение нужно проверить.")
    if d0 > 0 and d >= 0.9 * d0:
        return "Расхождение теоретических и измеренных значений приблизительно равно теоретически допустимой погрешности."
    return "Расхождение теоретических и измеренных значений не превосходит теоретически допустимую погрешность."


def default_conclusion(res: DCResult) -> str:
    q_, u = res.quantity, res.unit
    final_x = res.refine["x"] if res.refine else res.x_disp
    final_d = res.refine["delta"] if res.refine else res.delta
    final_d0 = res.refine["delta0"] if res.refine else res.delta0
    return (f"Рассчитанное значение {q_}0 = {q(res.x0, u)}, измеренное прибором с внутренним сопротивлением "
            f"{fmt_si(res.meter.r or 0, 'Ом')} — {q(final_x, u, 10)}. Расхождение Δ = {q(final_d, u, 4)}, "
            f"теоретически допустимая погрешность Δ0 = {q(final_d0, u, 4)}. " + final_phrase(res))


def debug_print(res: DCResult):
    print("Первое правило:", *[to_text(e) for e in res.k.kcl], sep="\n  ")
    print("Второе правило:", *[to_text(e) for e in res.k.kvl], sep="\n  ")
    print("Численно:", *[to_text(e) for e in res.k.kcl_num + res.k.kvl_num], sep="\n  ")
    print("Решение:", {k: num(v) for k, v in res.k.solution.items()})
    print(to_text(res.target_formula_num))
    print(text_summary(res))
    print(default_conclusion(res))


# ---------------------------------------------------------------- отчёт

def disp(x: float, unit: str, digits: int = DIGITS) -> str:
    """Показание прибора «как на индикаторе»: ровно digits значащих цифр, с приставкой: 21,22 В; 3,741 мА."""
    if x == 0:
        return f"0 {unit}"
    for mult, pref in ((1, ""), (1e-3, "м"), (1e-6, "мк"), (1e-9, "н")):
        if abs(x) >= mult * 0.99995:
            break
    v = x / mult
    dec = max(0, digits - 1 - int(math.floor(math.log10(abs(v)))))
    return config.dsep(f"{v:.{dec}f}").replace("-", "−") + f" {pref}{unit}"


def _paren(s: str) -> str:
    return f"({s})" if s.startswith("−") else s


def q(x: float, unit: str, digits: int = 5) -> str:
    """Величина с удобной приставкой, без лишних нулей: 0,00375 А → 3,75 мА."""
    if not math.isfinite(x):
        return "∞"
    return fmt_si(x, unit, digits).replace("-", "−")


def _target_calc(res: DCResult, ne: NodeEquations) -> list:
    """Формула искомой величины через потенциалы точек (вольтметр) или ток элемента (амперметр)."""
    m, Q, U = res.meter, res.quantity, res.unit
    pc, merge = physical(res.circuit)
    nm = ne.names
    if m.type == "VM":
        a, b = nm[merge.get(m.n1, m.n1)], nm[merge.get(m.n2, m.n2)]
        pa, pb = ne.potentials[a], ne.potentials[b]
        return [Sym(Q, "0"), " = ", Sym("φ", a), " − ", Sym("φ", b),
                f" = {num(pa)} − {_paren(num(pb))} = {q(res.x0, U)}"]
    el = series_element(res.circuit, m)
    if el is not None and el.name in ne.currents:
        val = ne.currents[el.name]
        sign = 1 if abs(val - res.x0) <= 1e-9 * max(1, abs(val)) else -1
        base = sym(el.name) if el.type == "I" else sym(el.name, "I")
        return [Sym(Q, "0"), " = ", "−" if sign < 0 else "", base, f" = {q(res.x0, U)}"]
    return [Sym(Q, "0"), f" = {q(res.x0, U)}"]


def report_blocks(res: DCResult, run_dir, engine: str, conclusion: str | None = None) -> list:
    """Блоки отчёта в порядке, рекомендованном преподавателем (п. 12 «Типичных ошибок»):
    задание (своя схема) → описание схемы → вычисление → описание процедуры измерений →
    измерение с результатом → расхождение → (дополнительные измерения) → теоретически допустимая
    погрешность → сравнение → вывод, который заканчивается обязательной фразой."""
    from pathlib import Path

    from .schematic import draw_schematic

    run_dir = Path(run_dir)
    m = res.meter
    is_v = m.type == "VM"
    Q, U = res.quantity, res.unit
    dev = "вольтметр" if is_v else "амперметр"
    dev_gen = "вольтметра" if is_v else "амперметра"
    dev_ins = "вольтметром" if is_v else "амперметром"
    rname = "V" if is_v else "A"
    ne = node_equations(res.circuit)
    pc, merge = physical(res.circuit)
    draw_schematic(pc, run_dir / "schematic_points.png", node_labels=ne.names)
    draw_schematic(res.circuit, run_dir / "schematic_measure.png",
                   readings={m.name: disp(res.x_disp, U)})

    blocks = [("bold", "Полученное задание"),
              ("figure", str(run_dir / "schematic.png"), "Схема для исследования"),
              ("bold", "Описание схемы")]
    for line in describe_circuit(res.circuit):
        blocks.append(("text", line))

    # ---- вычисление: правило узлов для каждой точки + закон Ома для каждого резистора
    blocks += [("bold", "Вычисление методом правил Кирхгофа"),
               ("text", "Для вычисления используется метод правил Кирхгофа: правило узлов для каждой точки "
                        "соединения и закон Ома для каждого резистора. Измерительный прибор при расчёте "
                        f"считается идеальным ({'RV = ∞' if is_v else 'RA = 0'}) и в схему не входит. "
                        "Точки соединения пронумерованы на схеме, потенциал точки 0 принят равным нулю; "
                        "токи обозначены буквой I с индексом элемента, ток в источнике напряжения "
                        "направлен от отрицательной клеммы к положительной."),
               ("figure", str(run_dir / "schematic_points.png"), "Схема с обозначением точек соединения"),
               ("text", "Потенциалы точек, к которым подключены источники напряжения:"),
               ("math", [System(ne.sources)]),
               ("text", "Правило узлов (сумма вытекающих из точки токов равна нулю):"),
               ("math", [System(ne.kcl)]),
               ("text", "Закон Ома для каждого резистора:"),
               ("math", [System(ne.ohm)]),
               ("text", "Подставляя значения (сопротивления — в омах, напряжения — в вольтах, токи — в амперах):"),
               ("math", [System(ne.sources_num + ne.kcl_num + ne.ohm_num)]),
               ("text", "Решив систему, получим потенциалы точек и токи:"),
               ("math", [System([[Sym("φ", k), f" = {q(v, 'В')}"] for k, v in sorted(ne.potentials.items(),
                                                                                   key=lambda t: int(t[0])) if k != "0"])]),
               ("math", [System([[sym(n) if n.startswith("J") else sym(n, "I"), f" = {q(v, 'А')}"]
                                 for n, v in ne.currents.items()])]),
               ("text", "Искомое значение:"),
               ("math", _target_calc(res, ne))]

    # ---- процедура измерений и измерение
    blocks += [("bold", "Описание процедуры измерений")]
    blocks += [("text", t) for t in measurement_text(res)]
    blocks += [("bold", "Измерение"),
               ("text", f"Цепь смоделирована {'программой ngspice' if engine == 'ngspice' else 'встроенным решателем'}; "
                        f"показание {dev_gen} приведено на схеме измерения."),
               ("figure", str(run_dir / "schematic_measure.png"), f"Измерение: показание {dev_gen} {disp(res.x_disp, U)}")]

    # ---- расхождение, теоретически допустимая погрешность, сравнение
    blocks += [("bold", "Расхождение теоретического и измеренного значений"),
               ("math", [Sym("Δ"), " = |", Sym(Q), " − ", Sym(Q, "0"), f"| = |{disp(res.x_disp, U)} − {_paren(q(res.x0, U))}|"
                         f" = {q(res.delta_raw, U, 4)}"])]
    if res.delta_raw < res.instr:
        blocks.append(("text", f"Полученное значение меньше инструментальной погрешности (половина единицы "
                               f"последнего разряда показания), поэтому принимаем Δ = {q(res.instr, U, 4)}."))
    blocks += [("bold", "Теоретически допустимая погрешность"),
               ("text", f"Относительная погрешность, вносимая {dev_ins}:"),
               ("math", [Sym("δ"), " = ", Frac([Sym("r")], [Sym("R", rname)]) if is_v
                         else Frac([Sym("R", rname)], [Sym("r")])]),
               ("text", f"где r — сопротивление схемы относительно точек подключения {dev_gen}. Чтобы найти r, "
                        f"{dev} исключается из схемы, источники напряжения заменяются проводами, источники тока — "
                        f"разрывами; сопротивление полученной пассивной схемы относительно точек подключения:"),
               ("math", [Sym("r"), f" = {q(res.r, 'Ом') if math.isfinite(res.r) else '∞'}"]),
               ("math", [Sym("δ"), " = ",
                         Frac([q(res.r, 'Ом')], [q(m.r or 0, 'Ом')]) if is_v
                         else Frac([q(m.r or 0, 'Ом')], [q(res.r, 'Ом')]), f" = {num(res.rel)}"]),
               ("math", [Sym("Δ", "0"), " = δ·|", Sym(Q), f"| = {num(res.rel)}·{disp(abs(res.x_disp), U)}"
                         f" = {q(res.delta0, U, 4)}"]),
               ("bold", "Сравнение")]
    if res.ok_first:
        blocks.append(("text", f"Расхождение Δ = {q(res.delta, U, 4)} не превосходит теоретически допустимую "
                               f"погрешность Δ0 = {q(res.delta0, U, 4)}."))
    else:
        blocks.append(("text", f"Расхождение Δ = {q(res.delta, U, 4)} превосходит теоретически допустимую "
                               f"погрешность Δ0 = {q(res.delta0, U, 4)}, поэтому необходимы дополнительные "
                               f"измерения — получение следующих цифр результата."))
    if res.refine:
        rf = res.refine
        draw_schematic(rf["circuit"], run_dir / "schematic_refine.png",
                       readings={m.name: disp(rf["x2"], U) if rf["x2"] else f"0 {U}"})
        if is_v:
            how = (f"последовательно с {dev_ins} подключается источник напряжения Eдоп = {disp(rf['x1'], U)}, "
                   f"равный первому показанию; внутреннее сопротивление учитывается резистором RV = {q(m.r, 'Ом')}, "
                   f"подключённым параллельно соединению {dev_gen} и источника, а сам {dev} считается идеальным")
        else:
            how = (f"параллельно {dev}у подключается источник тока Jдоп = {disp(rf['x1'], U)}, равный первому "
                   f"показанию; внутреннее сопротивление учитывается резистором RA = {q(m.r or 0, 'Ом')}, "
                   f"включённым последовательно с соединением {dev_gen} и источника, а сам {dev} считается идеальным")
        blocks += [("bold", "Дополнительные измерения"),
                   ("text", f"Для получения следующих цифр результата {how}."),
                   ("figure", str(run_dir / "schematic_refine.png"), "Схема дополнительного измерения"),
                   ("text", f"Показания {dev_gen}: {Q}2 = {disp(rf['x2'], U) if rf['x2'] else '0 ' + U}."),
                   ("math", [Sym(Q), " = ", Sym(Q, "1"), " + ", Sym(Q, "2"),
                             f" = {disp(rf['x1'], U)} + {_paren(disp(rf['x2'], U)) if rf['x2'] else '0'} = {q(rf['x'], U, 10)}"]),
                   ("math", [Sym("Δ"), " = |", Sym(Q), " − ", Sym(Q, "0"), f"| = {q(rf['delta'], U, 4)}"]),
                   ("text", "Теоретически допустимая погрешность по уточнённому результату:"),
                   ("math", [Sym("Δ", "0"), " = δ·|", Sym(Q), f"| = {num(res.rel)}·{q(abs(rf['x']), U, 10)}"
                             f" = {q(rf['delta0'], U, 4)}"]),
                   ("text", f"Расхождение Δ = {q(rf['delta'], U, 4)} "
                            + ("не превосходит" if rf["ok"] else "превосходит")
                            + f" теоретически допустимую погрешность Δ0 = {q(rf['delta0'], U, 4)}.")]
    concl = (conclusion or default_conclusion(res)).strip()
    if not concl.endswith(final_phrase(res)):
        concl = concl.rstrip(".") + ". " + final_phrase(res)
    blocks += [("bold", "Вывод"), ("text", concl)]
    return blocks
