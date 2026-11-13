"""Описание электрической цепи.

Это тот JSON, который составляет языковая модель. Из него же делается
и симуляция (ngspice или встроенный решатель), и картинка схемы,
поэтому схема на рисунке всегда совпадает с тем, что посчитано.

Пример:
{
  "title": "Делитель напряжения",
  "elements": [
    {"name": "E1",  "type": "V",  "n1": "1", "n2": "0", "value": 10},
    {"name": "PA1", "type": "AM", "n1": "1", "n2": "2", "r": 1},
    {"name": "R1",  "type": "R",  "n1": "2", "n2": "3", "value": "100k"},
    {"name": "R2",  "type": "R",  "n1": "3", "n2": "0", "value": "100k"},
    {"name": "PV1", "type": "VM", "n1": "3", "n2": "0", "r": "1M"}
  ],
  "analyses": [{"type": "op"}]
}

Узел "0" — общий провод (земля). Типы элементов:
    R  резистор, Ом            C  конденсатор, Ф          L  катушка, Гн
    V  источник ЭДС, В         I  источник тока, А
    AM амперметр (r — внутреннее сопротивление, Ом; 0 или нет — идеальный)
    VM вольтметр (r — входное сопротивление, Ом; нет — идеальный)

Для V: n1 — плюс. Показание VM = U(n1) − U(n2). Показание AM — ток от n1 к n2.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from . import config

GROUND = "0"
_GROUND_ALIASES = {"0", "gnd", "ground", "земля", "общий", "com"}

# Множители: латиница (как в SPICE) и кириллица (как пишут в методичках)
_PREFIXES = {
    "t": 1e12, "g": 1e9, "meg": 1e6, "м": 1e-3, "мк": 1e-6, "u": 1e-6, "µ": 1e-6, "μ": 1e-6,
    "n": 1e-9, "н": 1e-9, "p": 1e-12, "п": 1e-12, "f": 1e-15, "k": 1e3, "к": 1e3,
    "m": 1e-3, "mil": 25.4e-6, "г": 1e9, "т": 1e12,
}
_UNITS = ("ом", "ohm", "ω", "ф", "f", "гн", "h", "в", "v", "а", "a", "гц", "hz", "с", "s")
_NUM_RE = re.compile(r"^\s*([-+]?\d+(?:[.,]\d*)?(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


def parse_value(raw: Union[str, float, int, None]) -> Optional[float]:
    """'4.7k' → 4700, '10 мкФ' → 1e-5, '1M' → 1e6 (M = мега, как в обычной записи), 0.1 → 0.1."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    m = _NUM_RE.match(s)
    if not m:
        raise ValueError(f"не понимаю значение '{raw}'")
    number = float(m.group(1).replace(",", "."))
    rest = m.group(2).replace(" ", "").rstrip(".")
    return float(f"{number * _multiplier(rest, raw):.12g}")


def _multiplier(rest: str, raw) -> float:
    if not rest:
        return 1.0

    def is_unit(tail: str) -> bool:
        return tail == "" or tail.lower() in _UNITS

    # 1) только единица измерения: «5 В», «1F», «10 Ом»
    if is_unit(rest):
        return 1.0
    # 2) «мега»: заглавная M/М (как пишут в методичках: 1 МОм, 1M) или SPICE-овское meg
    if rest.lower().startswith("meg") and is_unit(rest[3:]):
        return 1e6
    if rest[0] in ("M", "М") and rest[:2].lower() != "мк" and is_unit(rest[1:]):
        return 1e6
    # 3) обычные приставки: k, к, m, м, мк, u, n, н, p, п ...
    low = rest.lower()
    for pref in sorted(_PREFIXES, key=len, reverse=True):
        if low.startswith(pref) and is_unit(rest[len(pref):]):
            return _PREFIXES[pref]
    raise ValueError(f"не понимаю множитель '{rest}' в значении '{raw}'")


ElementType = Literal["R", "C", "L", "V", "I", "AM", "VM"]


class Element(BaseModel):
    name: str = Field(description="Позиционное обозначение по ГОСТ: R1, C1, L1, E1, J1, PA1, PV1")
    type: ElementType
    n1: str
    n2: str
    value: Optional[float] = Field(None, description="Номинал в СИ: Ом, Ф, Гн, В, А")
    r: Optional[float] = Field(None, description="Внутреннее сопротивление прибора, Ом")
    ac: Optional[float] = Field(None, description="Амплитуда для частотного анализа (обычно 1)")
    wave: Literal["dc", "step", "sine"] = Field("dc", description="Форма сигнала источника во времени")
    amplitude: Optional[float] = None
    freq: Optional[float] = None
    phase: float = 0.0

    @field_validator("value", "r", "ac", "amplitude", "freq", mode="before")
    @classmethod
    def _parse(cls, v):
        return parse_value(v)

    @field_validator("type", mode="before")
    @classmethod
    def _type(cls, v):
        v = str(v).strip().upper()
        aliases = {"E": "V", "VS": "V", "VSRC": "V", "J": "I", "IS": "I", "PA": "AM", "A": "AM",
                   "AMMETER": "AM", "PV": "VM", "VOLTMETER": "VM", "RESISTOR": "R",
                   "CAPACITOR": "C", "INDUCTOR": "L"}
        return aliases.get(v, v)

    @field_validator("n1", "n2", mode="before")
    @classmethod
    def _node(cls, v):
        s = str(v).strip()
        return GROUND if s.lower() in _GROUND_ALIASES else s

    @model_validator(mode="after")
    def _check(self):
        if self.n1 == self.n2:
            raise ValueError(f"{self.name}: оба вывода подключены к одному узлу '{self.n1}'")
        if self.type in ("R", "C", "L"):
            if self.value is None or self.value <= 0:
                raise ValueError(f"{self.name}: нужен положительный номинал value")
        if self.type in ("V", "I") and self.value is None and self.amplitude is None and self.ac is None:
            self.value = 0.0
        if self.type == "VM" and self.r is not None and self.r <= 0:
            raise ValueError(f"{self.name}: входное сопротивление вольтметра должно быть > 0 (или не указано = идеальный)")
        if self.type == "AM" and self.r is not None and self.r < 0:
            raise ValueError(f"{self.name}: сопротивление амперметра не может быть отрицательным")
        if self.wave == "sine" and not self.freq:
            raise ValueError(f"{self.name}: для синусоиды нужна частота freq")
        return self

    @property
    def is_meter(self) -> bool:
        return self.type in ("AM", "VM")

    @property
    def is_source(self) -> bool:
        return self.type in ("V", "I")


class Analysis(BaseModel):
    type: Literal["op", "ac", "tran"]
    # ac: частотный анализ
    f_start: Optional[float] = None
    f_stop: Optional[float] = None
    points_per_decade: int = 20
    # tran: переходный процесс
    t_stop: Optional[float] = None
    t_step: Optional[float] = None

    @field_validator("type", mode="before")
    @classmethod
    def _type(cls, v):
        v = str(v).strip().lower()
        return {"dc": "op", "operating_point": "op", "transient": "tran", "freq": "ac"}.get(v, v)

    @field_validator("f_start", "f_stop", "t_stop", "t_step", mode="before")
    @classmethod
    def _parse(cls, v):
        return parse_value(v)

    @model_validator(mode="after")
    def _check(self):
        if self.type == "ac":
            self.f_start = self.f_start or 10.0
            self.f_stop = self.f_stop or 100e3
            if self.f_stop <= self.f_start:
                raise ValueError("ac: f_stop должна быть больше f_start")
        if self.type == "tran":
            if not self.t_stop or self.t_stop <= 0:
                raise ValueError("tran: нужно время моделирования t_stop > 0")
            self.t_step = self.t_step or self.t_stop / 1000
        return self


def dangling_problem(circuit) -> str | None:
    """Узлы, к которым подключён только один вывод (обрыв). Текст — замечание для модели."""
    touch: dict[str, list] = {}
    for e in circuit.elements:
        if e.type == "VM":
            continue
        for n in (e.n1, e.n2):
            touch.setdefault(n, []).append(e)
    bad = [(n, es[0]) for n, es in touch.items() if len(es) < 2]
    if not bad:
        return None
    parts = [f"к точке «{n}» подключён только {e.name} (его выводы: {e.n1} и {e.n2})" for n, e in bad]
    return ("в цепи обрыв: " + "; ".join(parts) + ". В замкнутой цепи к каждой точке подключено минимум два "
            "вывода. Посмотри на картинку: с каким элементом соединён этот вывод проводом? Дай этому выводу "
            "тот же номер точки, что и у того элемента (точки, соединённые проводом, — одна точка)")


class Circuit(BaseModel):
    title: str = "Исследуемая цепь"
    elements: list[Element]
    analyses: list[Analysis] = Field(default_factory=lambda: [Analysis(type="op")])

    @model_validator(mode="after")
    def _check(self):
        problems = []
        names = Counter(e.name for e in self.elements)
        dup = [n for n, k in names.items() if k > 1]
        if dup:
            problems.append(f"повторяются имена элементов: {', '.join(dup)}")
        if not any(e.is_source for e in self.elements):
            problems.append("в цепи нет ни одного источника (V или I)")
        nodes = Counter()
        for e in self.elements:
            if e.type == "VM":  # вольтметр не замыкает цепь, узлы считаем без него
                continue
            nodes[e.n1] += 1
            nodes[e.n2] += 1
        if GROUND not in nodes:
            problems.append('нет общего узла "0" (земли)')
        # оборванные узлы — не жёсткая ошибка: см. dangling_problem (мягкая проверка, «хоть криво»)
        for e in self.elements:
            if e.type == "VM":
                for n in (e.n1, e.n2):
                    if n not in nodes:
                        problems.append(f"{e.name} подключён к узлу '{n}', которого нет в цепи")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    # --- удобные методы
    def nodes(self) -> list[str]:
        seen: list[str] = []
        for e in self.elements:
            for n in (e.n1, e.n2):
                if n != GROUND and n not in seen:
                    seen.append(n)
        return seen

    def meters(self) -> list[Element]:
        return [e for e in self.elements if e.is_meter]

    def idealized(self) -> "Circuit":
        """Та же цепь с идеальными приборами — для расчёта методической погрешности."""
        c = self.model_copy(deep=True)
        for e in c.elements:
            if e.type == "VM":
                e.r = None
            elif e.type == "AM":
                e.r = 0.0
        return c

    def has_real_meters(self) -> bool:
        return any((e.type == "VM" and e.r) or (e.type == "AM" and e.r) for e in self.elements)


def repair_circuit(data: Union[dict, str]) -> tuple[Circuit, list[str]]:
    """Последний шанс «хоть как-то»: из описания, которое модель так и не исправила, сделать
    считаемую цепь — выкинуть элементы без номинала и висящие ветви, назначить общий провод.
    Возвращает (цепь, что исправлено). Если не вышло — ValueError."""
    import json

    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict) and "circuit" in data and "elements" not in data:
        data = data["circuit"]
    notes: list[str] = []
    els = []
    for raw in (data or {}).get("elements", []):
        try:
            els.append(Element.model_validate(raw))
        except Exception:
            notes.append(f"выброшен элемент {raw.get('name', '?') if isinstance(raw, dict) else raw}: неполное описание")
    changed = True
    while changed:          # висящие ветви — по одной, пока не останется замкнутая цепь
        changed = False
        deg = Counter(n for e in els for n in (e.n1, e.n2))
        for e in list(els):
            if (deg[e.n1] < 2 or deg[e.n2] < 2) and not e.is_meter:
                els.remove(e)
                notes.append(f"выброшен {e.name}: висел в воздухе")
                changed = True
                break
    deg = Counter(n for e in els for n in (e.n1, e.n2))
    if els and GROUND not in deg:
        top = deg.most_common(1)[0][0]
        for e in els:
            e.n1, e.n2 = (GROUND if e.n1 == top else e.n1), (GROUND if e.n2 == top else e.n2)
        notes.append(f"общим проводом назначена точка {top}")
    c = Circuit.model_validate({"title": (data or {}).get("title", "Цепь") + " (исправлена автоматически)",
                                "elements": [e.model_dump() for e in els],
                                "analyses": (data or {}).get("analyses") or [{"type": "op"}]})
    return c, notes


def load_circuit(data) -> Circuit:
    """Принимает Circuit, dict, JSON-строку (модели иногда присылают JSON строкой)
    или путь к .json-файлу (модель в песочнице часто пишет load_circuit('/data/circuit.json'))."""
    import json
    from pathlib import Path

    if isinstance(data, Circuit):
        return data
    if isinstance(data, Path) or (isinstance(data, str) and not data.lstrip().startswith(("{", "["))
                                  and data.strip().endswith(".json")):
        data = Path(str(data).strip()).read_text(encoding="utf-8")
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict) and "circuit" in data and "elements" not in data:
        data = data["circuit"]
    return Circuit.model_validate(data)


# --- форматирование величин для подписей (по-русски)
_UNIT_RU = {"R": "Ом", "C": "Ф", "L": "Гн", "V": "В", "I": "А", "AM": "Ом", "VM": "Ом"}


def fmt_si(x: float, unit: str, digits: int = 3) -> str:
    """4700, 'Ом' → '4,7 кОм'; 1e-7, 'Ф' → '0,1 мкФ'."""
    if x == 0:
        return f"0 {unit}"
    steps = [(1e9, "Г"), (1e6, "М"), (1e3, "к"), (1, ""), (1e-3, "м"), (1e-6, "мк"), (1e-9, "н"), (1e-12, "п")]
    ax = abs(x)
    for mult, pref in steps:
        if ax >= mult * 0.9999:
            break
    val = x / mult
    if abs(val) >= 1000 and pref != "Г":  # 0,1 мкФ лучше, чем 100 нФ? оставляем как есть
        pass
    s = config.dsep(f"{val:.{digits}g}")
    return f"{s} {pref}{unit}"


def fmt_r_ru(x: float) -> str:
    """Номинал резистора на схеме по российским правилам: Ом — только число, кОм — «к», МОм — «М»:
    500, 1 к, 4.7 к, 2 М."""
    for mult, suf in ((1e6, " М"), (1e3, " к"), (1, "")):
        if abs(x) >= mult * 0.9999:
            return config.dsep(f"{x / mult:.4g}") + suf
    return config.dsep(f"{x:.4g}")


def element_value_label(e: Element) -> str:
    ru = config.SCHEMATIC_LABELS == "bmstu"
    if e.type == "R" and ru:
        return fmt_r_ru(e.value)
    if e.type in ("R", "C", "L"):
        return fmt_si(e.value, _UNIT_RU[e.type])
    if e.type in ("V", "I"):
        if e.wave == "sine":
            return f"{fmt_si(e.amplitude or e.value or 0, _UNIT_RU[e.type])}, {fmt_si(e.freq, 'Гц')}"
        if e.value:
            return fmt_si(e.value, _UNIT_RU[e.type])
        return ""
    if e.type in ("AM", "VM") and e.r:
        return fmt_r_ru(e.r) if ru else fmt_si(e.r, "Ом")
    return ""
