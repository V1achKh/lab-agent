"""Встроенный решатель линейных цепей (модифицированный метод узловых потенциалов, MNA).

Зачем он, если есть ngspice:
  * проект запускается даже без установленного ngspice;
  * можно сверить результаты двух независимых движков (хороший пункт для отчёта);
  * код короткий, и на защите его можно объяснить: это обычные законы Кирхгофа в матричном виде.

Неизвестные: потенциалы узлов (кроме земли) и токи «ветвей с напряжением»
(источники ЭДС, амперметры, катушки). Для каждой такой ветви пишется уравнение
    U(n1) − U(n2) − Z·I = E,
где для источника Z = 0, для амперметра Z = r, для катушки Z = jωL.
Переходный процесс считается методом трапеций (первые шаги — неявным Эйлером).
"""
from __future__ import annotations

import math

import numpy as np

from .circuit import GROUND, Analysis, Circuit, Element


class _Index:
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.node = {n: i for i, n in enumerate(circuit.nodes())}
        self.n_nodes = len(self.node)
        # ветви, для которых ток — отдельная неизвестная
        self.branch = {}
        for e in circuit.elements:
            if e.type in ("V", "AM", "L"):
                self.branch[e.name] = self.n_nodes + len(self.branch)
        self.size = self.n_nodes + len(self.branch)

    def ni(self, node: str):
        return None if node == GROUND else self.node[node]


def _stamp_g(A, i, j, g):
    if i is not None:
        A[i, i] += g
    if j is not None:
        A[j, j] += g
    if i is not None and j is not None:
        A[i, j] -= g
        A[j, i] -= g


def _stamp_branch(A, b, idx: _Index, e: Element, z, emf):
    k = idx.branch[e.name]
    i, j = idx.ni(e.n1), idx.ni(e.n2)
    if i is not None:
        A[i, k] += 1
        A[k, i] += 1
    if j is not None:
        A[j, k] -= 1
        A[k, j] -= 1
    A[k, k] -= z
    b[k] += emf


def _stamp_current(b, idx: _Index, e: Element, value):
    # ток источника течёт от n1 через источник к n2 (как в SPICE)
    i, j = idx.ni(e.n1), idx.ni(e.n2)
    if i is not None:
        b[i] -= value
    if j is not None:
        b[j] += value


def _solve(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "система уравнений вырождена: проверьте, что каждый узел связан с землёй "
            "(нет висящих узлов, контуров из одних источников ЭДС или сечений из одних источников тока)"
        ) from exc


def _readings_from_x(circuit: Circuit, idx: _Index, x) -> dict:
    def u(node):
        return 0.0 if node == GROUND else x[idx.node[node]]

    out = {}
    for e in circuit.elements:
        if e.type == "VM":
            out[e.name] = u(e.n1) - u(e.n2)
        elif e.type == "AM":
            out[e.name] = x[idx.branch[e.name]]
    return out


def _source_value(e: Element, t: float | None) -> float:
    """Значение источника в момент t (t=None — постоянная составляющая)."""
    if t is None:
        return e.value or 0.0
    if e.wave == "dc":
        return e.value or 0.0
    if e.wave == "step":
        amp = e.amplitude if e.amplitude is not None else (e.value or 0.0)
        return amp if t > 0 else 0.0
    amp = e.amplitude if e.amplitude is not None else (e.value or 0.0)
    offset = e.value if (e.amplitude is not None and e.value) else 0.0
    return offset + amp * math.sin(2 * math.pi * e.freq * t + math.radians(e.phase))


def _build(circuit: Circuit, omega: float, ac: bool):
    idx = _Index(circuit)
    dtype = complex if ac else float
    A = np.zeros((idx.size, idx.size), dtype=dtype)
    b = np.zeros(idx.size, dtype=dtype)
    first_v = next((e.name for e in circuit.elements if e.is_source), None)
    any_ac = any(e.ac for e in circuit.elements)
    for e in circuit.elements:
        i, j = idx.ni(e.n1), idx.ni(e.n2)
        if e.type == "R":
            _stamp_g(A, i, j, 1.0 / e.value)
        elif e.type == "C":
            if ac:
                _stamp_g(A, i, j, 1j * omega * e.value)
        elif e.type == "VM":
            if e.r:
                _stamp_g(A, i, j, 1.0 / e.r)
        elif e.type == "L":
            _stamp_branch(A, b, idx, e, (1j * omega * e.value) if ac else 0.0, 0.0)
        elif e.type == "AM":
            _stamp_branch(A, b, idx, e, e.r or 0.0, 0.0)
        elif e.type == "V":
            if ac:
                emf = e.ac if any_ac else (1.0 if e.name == first_v else 0.0)
                emf = emf or 0.0
            else:
                emf = _source_value(e, None)
            _stamp_branch(A, b, idx, e, 0.0, emf)
        elif e.type == "I":
            if ac:
                val = e.ac if any_ac else (1.0 if e.name == first_v else 0.0)
                val = val or 0.0
            else:
                val = _source_value(e, None)
            _stamp_current(b, idx, e, val)
    # крошечная проводимость на землю, чтобы узлы за идеальным вольтметром не «висели»
    for k in range(idx.n_nodes):
        A[k, k] += 1e-15
    return idx, A, b


def operating_point(circuit) -> dict:
    """Режим постоянного тока. circuit — Circuit, dict, JSON-строка или путь к .json."""
    if not isinstance(circuit, Circuit):
        from .circuit import load_circuit

        circuit = load_circuit(circuit)
    idx, A, b = _build(circuit, 0.0, ac=False)
    x = _solve(A, b)
    nodes = {n: float(x[i]) for n, i in idx.node.items()}

    def u(node):
        return 0.0 if node == GROUND else nodes[node]

    # токи всех элементов, положительное направление — от n1 к n2 через элемент
    currents = {}
    for e in circuit.elements:
        if e.type == "R":
            currents[e.name] = (u(e.n1) - u(e.n2)) / e.value
        elif e.type in ("V", "AM", "L"):
            currents[e.name] = float(x[idx.branch[e.name]])
        elif e.type == "I":
            currents[e.name] = _source_value(e, None)
        elif e.type == "VM":
            currents[e.name] = (u(e.n1) - u(e.n2)) / e.r if e.r else 0.0
        else:
            currents[e.name] = 0.0
    return {"nodes": nodes, "currents": currents,
            "readings": {k: float(v) for k, v in _readings_from_x(circuit, idx, x).items()}}


def thevenin_resistance(circuit: Circuit, a: str, b: str, remove: tuple = ()) -> float:
    """Сопротивление схемы относительно точек a и b: источники ЭДС закорочены,
    источники тока разорваны, элементы из remove (обычно сам прибор) убраны.
    Считаем «по-честному»: подаём тестовый ток 1 А и смотрим напряжение."""
    from .circuit import Element

    els = []
    for e in circuit.elements:
        if e.name in remove or e.type == "I":
            continue
        if e.type == "VM" and not e.r:
            continue
        e = e.model_copy()
        if e.type == "V":
            e.value, e.wave, e.amplitude, e.ac = 0.0, "dc", None, None
        els.append(e)
    els.append(Element(name="Jtest", type="I", n1=b, n2=a, value=1.0))
    if not any(e.type == "V" for e in els):
        els.append(Element(name="Vref", type="V", n1="0", n2="0x", value=0.0))
        els.append(Element(name="Rref", type="R", n1="0x", n2="0", value=1.0))
    test = Circuit.model_construct(title="r", elements=els, analyses=[])
    idx, A, rhs = _build(test, 0.0, ac=False)
    x = _solve(A, rhs)

    def u(node):
        return 0.0 if node == GROUND else x[idx.node[node]]

    r = float(u(a) - u(b))
    return float("inf") if r > 1e10 else max(r, 0.0)


def ac_sweep(circuit: Circuit, an: Analysis) -> dict:
    decades = math.log10(an.f_stop / an.f_start)
    n = max(2, int(round(decades * an.points_per_decade)) + 1)
    freqs = np.logspace(math.log10(an.f_start), math.log10(an.f_stop), n)
    readings: dict[str, list[complex]] = {m.name: [] for m in circuit.meters()}
    for f in freqs:
        idx, A, b = _build(circuit, 2 * math.pi * f, ac=True)
        x = _solve(A, b)
        for k, v in _readings_from_x(circuit, idx, x).items():
            readings[k].append(complex(v))
    return {"freq": freqs.tolist(), "readings": readings}


def transient(circuit: Circuit, an: Analysis) -> dict:
    """Метод трапеций с нулевыми начальными условиями (как .tran ... uic в SPICE)."""
    h = min(an.t_step, an.t_stop / 200)
    steps = int(math.ceil(an.t_stop / h))
    if steps > 200_000:
        raise RuntimeError("слишком мелкий шаг t_step для такого t_stop")
    idx = _Index(circuit)
    caps = [e for e in circuit.elements if e.type == "C"]
    inds = [e for e in circuit.elements if e.type == "L"]
    v_c = {e.name: 0.0 for e in caps}
    i_c = {e.name: 0.0 for e in caps}
    v_l = {e.name: 0.0 for e in inds}
    i_l = {e.name: 0.0 for e in inds}
    times = [0.0]
    readings = {m.name: [0.0] for m in circuit.meters()}

    def u(x, node):
        return 0.0 if node == GROUND else x[idx.node[node]]

    for step in range(1, steps + 1):
        t = min(step * h, an.t_stop)
        dt = t - times[-1]
        euler = step <= 2  # первые шаги — неявный Эйлер, чтобы не было «звона» трапеций
        A = np.zeros((idx.size, idx.size))
        b = np.zeros(idx.size)
        for e in circuit.elements:
            i, j = idx.ni(e.n1), idx.ni(e.n2)
            if e.type == "R":
                _stamp_g(A, i, j, 1.0 / e.value)
            elif e.type == "VM":
                if e.r:
                    _stamp_g(A, i, j, 1.0 / e.r)
            elif e.type == "C":
                g = (e.value / dt) if euler else (2 * e.value / dt)
                ieq = g * v_c[e.name] + (0.0 if euler else i_c[e.name])
                _stamp_g(A, i, j, g)
                # i_new = g·v_new − ieq  →  источник тока ieq от n2 к n1
                if i is not None:
                    b[i] += ieq
                if j is not None:
                    b[j] -= ieq
            elif e.type == "L":
                if euler:
                    z = e.value / dt
                    emf = -z * i_l[e.name]
                else:
                    z = 2 * e.value / dt
                    emf = -z * i_l[e.name] - v_l[e.name]
                _stamp_branch(A, b, idx, e, z, emf)
            elif e.type == "AM":
                _stamp_branch(A, b, idx, e, e.r or 0.0, 0.0)
            elif e.type == "V":
                _stamp_branch(A, b, idx, e, 0.0, _source_value(e, t))
            elif e.type == "I":
                _stamp_current(b, idx, e, _source_value(e, t))
        for k in range(idx.n_nodes):
            A[k, k] += 1e-15
        x = _solve(A, b)
        for e in caps:
            v_new = u(x, e.n1) - u(x, e.n2)
            if euler:
                i_c[e.name] = e.value / dt * (v_new - v_c[e.name])
            else:
                i_c[e.name] = 2 * e.value / dt * (v_new - v_c[e.name]) - i_c[e.name]
            v_c[e.name] = v_new
        for e in inds:
            i_l[e.name] = float(x[idx.branch[e.name]])
            v_l[e.name] = u(x, e.n1) - u(x, e.n2)
        times.append(t)
        for k, v in _readings_from_x(circuit, idx, x).items():
            readings[k].append(float(v))
    return {"time": times, "readings": readings}


def run(circuit: Circuit, an: Analysis) -> dict:
    if an.type == "op":
        return operating_point(circuit)
    if an.type == "ac":
        return ac_sweep(circuit, an)
    return transient(circuit, an)
