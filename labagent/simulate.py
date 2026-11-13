"""Общий интерфейс симуляции: выбор движка, расчёт с идеальными приборами,
текстовая сводка результатов для языковой модели и таблицы для отчёта."""
from __future__ import annotations

import cmath
import logging
import math
from pathlib import Path

import numpy as np

from . import config, solver, spice
from .circuit import Circuit, fmt_si

log = logging.getLogger(__name__)


def pick_engine(engine: str | None = None) -> str:
    engine = engine or config.SIM_ENGINE
    if engine == "auto":
        return "ngspice" if spice.available() else "builtin"
    return engine


def _run_one(circuit: Circuit, an, engine: str, keep_dir: Path | None):
    if engine == "ngspice":
        try:
            return spice.run(circuit, an, keep_dir=keep_dir), "ngspice"
        except Exception as exc:  # в auto-режиме не падаем, а считаем сами
            if config.SIM_ENGINE == "ngspice":
                raise
            log.warning("ngspice не справился (%s), считаю встроенным решателем", exc)
    return solver.run(circuit, an), "builtin"


def simulate(circuit: Circuit, engine: str | None = None, keep_dir: Path | None = None,
             with_ideal: bool = True) -> dict:
    engine = pick_engine(engine)
    out = {"title": circuit.title, "engine": engine, "analyses": {}}
    for an in circuit.analyses:
        res, used = _run_one(circuit, an, engine, keep_dir)
        res["settings"] = an.model_dump()
        out["analyses"][an.type] = res
        out["engine"] = used
    # методическая погрешность: та же цепь с идеальными приборами
    if with_ideal and circuit.has_real_meters():
        from .circuit import Analysis

        ideal_c = circuit.idealized()
        res, _ = _run_one(ideal_c, Analysis(type="op"), engine, None)
        out["ideal_op"] = res
    return out


# ---------------------------------------------------------------- сводки

def meter_unit(circuit: Circuit, name: str) -> str:
    e = next(e for e in circuit.elements if e.name == name)
    return "В" if e.type == "VM" else "А"


def cutoff_frequency(freq: list[float], values: list[complex]) -> float | None:
    """Частота среза по уровню 0,707 от значения на нижней частоте (−3 дБ)."""
    mags = np.abs(np.array(values))
    if len(mags) < 2 or mags[0] == 0:
        return None
    level = mags[0] / math.sqrt(2)
    for k in range(1, len(mags)):
        if (mags[k - 1] - level) * (mags[k] - level) <= 0 and mags[k - 1] != mags[k]:
            # линейная интерполяция в логарифме частоты
            x0, x1 = math.log10(freq[k - 1]), math.log10(freq[k])
            y0, y1 = mags[k - 1], mags[k]
            x = x0 + (level - y0) * (x1 - x0) / (y1 - y0)
            return 10 ** x
    return None


def time_constant(time: list[float], values: list[float]) -> float | None:
    """Оценка постоянной времени τ: время, за которое отклонение от установившегося
    значения уменьшается до 36,8 % (для заряда это те же 63,2 % от конечного значения).
    Отсчёт ведём от момента наибольшего отклонения: так работает и для затухающего тока."""
    v = np.array(values)
    final = v[-1]
    dev = np.abs(v - final)
    p = int(np.argmax(dev))
    if dev[p] < 1e-12 or p >= len(v) - 2:
        return None
    target = final + 0.368 * (v[p] - final)
    for k in range(p + 1, len(v)):
        if (v[k - 1] - target) * (v[k] - target) <= 0 and v[k] != v[k - 1]:
            t0, t1 = time[k - 1], time[k]
            return t0 + (target - v[k - 1]) * (t1 - t0) / (v[k] - v[k - 1]) - time[p]
    return None


def clean(x: float) -> float:
    """Убираем численный «шум» вроде 1e-11 А (следствие gmin в решателе)."""
    return 0.0 if abs(x) < 1e-10 else x


_PREF = [(1e3, "к"), (1, ""), (1e-3, "м"), (1e-6, "мк"), (1e-9, "н")]


def col_scale(values, unit: str) -> tuple[float, str]:
    """Общий множитель для столбца таблицы: 0,0003 А → (1e3, 'мА')."""
    m = max((abs(v) for v in values), default=0.0)
    if m == 0 or not unit:
        return 1.0, unit
    for mult, pref in _PREF:
        if m >= mult * 0.9999:
            return 1 / mult, pref + unit
    return 1e9, "н" + unit


def _num(x: float, digits: int = 4) -> str:
    return config.dsep(f"{x:.{digits}g}")


def method_errors(circuit: Circuit, sim: dict) -> list[dict]:
    """Таблица: показание реального прибора, идеального и методическая погрешность."""
    if "ideal_op" not in sim or "op" not in sim["analyses"]:
        return []
    rows = []
    real = sim["analyses"]["op"]["readings"]
    ideal = sim["ideal_op"]["readings"]
    for m in circuit.meters():
        r, i = real.get(m.name), ideal.get(m.name)
        if r is None or i is None:
            continue
        r, i = clean(r), clean(i)
        # если идеальный прибор показывает ~0, относительная погрешность не имеет смысла
        meaningful = abs(i) > 1e-12 and abs(i) >= 1e-3 * abs(r)
        err = (r - i) / i * 100 if meaningful else float("nan")
        rows.append({"meter": m.name, "unit": meter_unit(circuit, m.name),
                     "real": r, "ideal": i, "error_pct": err, "r": m.r})
    return rows


def summary_text(circuit: Circuit, sim: dict) -> str:
    """Короткий текст для языковой модели: только главное, чтобы не забивать контекст."""
    lines = [f"Цепь: {circuit.title}. Движок: {sim['engine']}."]
    an = sim["analyses"]
    if "op" in an:
        lines.append("Постоянный ток, показания приборов:")
        for name, val in an["op"]["readings"].items():
            lines.append(f"  {name} = {fmt_si(clean(val), meter_unit(circuit, name), 4)}")
        if not an["op"]["readings"]:
            lines.append("  (в цепи нет приборов; потенциалы узлов: "
                         + ", ".join(f"{n}: {fmt_si(v, 'В', 4)}" for n, v in an["op"]["nodes"].items()) + ")")
        for row in method_errors(circuit, sim):
            if math.isnan(row["error_pct"]):
                continue
            lines.append(
                f"  {row['meter']}: если бы все приборы были идеальными — {fmt_si(row['ideal'], row['unit'], 4)}, "
                f"отклонение из-за сопротивлений приборов {row['error_pct']:+.2f} %")
    if "ac" in an:
        a = an["ac"]
        lines.append(f"Частотный анализ {fmt_si(a['freq'][0], 'Гц')} … {fmt_si(a['freq'][-1], 'Гц')}:")
        for name, vals in a["readings"].items():
            fc = cutoff_frequency(a["freq"], vals)
            unit = meter_unit(circuit, name)
            m0, m1 = abs(vals[0]), abs(vals[-1])
            ph0, ph1 = (math.degrees(cmath.phase(v)) for v in (vals[0], vals[-1]))
            txt = (f"  {name} (при амплитуде источника 1): на нижней частоте {fmt_si(m0, unit, 4)}, фаза {ph0:.1f}°; "
                   f"на верхней {fmt_si(m1, unit, 4)}, фаза {ph1:.1f}°")
            if fc:
                txt += f"; частота среза (−3 дБ) ≈ {fmt_si(fc, 'Гц', 4)}"
            lines.append(txt)
    if "tran" in an:
        t = an["tran"]
        lines.append(f"Переходный процесс 0 … {fmt_si(t['time'][-1], 'с')}:")
        for name, vals in t["readings"].items():
            tau = time_constant(t["time"], vals)
            unit = meter_unit(circuit, name)
            peak = max(vals, key=abs)
            txt = f"  {name}: в начале {fmt_si(clean(vals[0]), unit, 4)}, в конце {fmt_si(clean(vals[-1]), unit, 4)}"
            if abs(peak) > 1.05 * max(abs(vals[0]), abs(vals[-1])):
                txt += f", максимум {fmt_si(peak, unit, 4)}"
            if tau:
                txt += f", постоянная времени по уровню 63,2 % ≈ {fmt_si(tau, 'с', 4)}"
            lines.append(txt)
    return "\n".join(lines)


def tables(circuit: Circuit, sim: dict) -> list[dict]:
    """Таблицы для отчёта: [{'caption', 'headers', 'rows'}]."""
    out = []
    an = sim["analyses"]
    if "op" in an and an["op"]["readings"]:
        rows = []
        errs = {r["meter"]: r for r in method_errors(circuit, sim)}
        for m in circuit.meters():
            val = clean(an["op"]["readings"][m.name])
            unit = meter_unit(circuit, m.name)
            row = [m.name, "Напряжение" if m.type == "VM" else "Ток", fmt_si(val, unit, 4)]
            if errs:
                e = errs.get(m.name)
                if e is None:
                    row += ["—", "—"]
                else:
                    pct = "—" if math.isnan(e["error_pct"]) else config.dsep(f"{e['error_pct']:+.2f}")
                    row += [fmt_si(e["ideal"], unit, 4), pct]
            rows.append(row)
        headers = ["Прибор", "Величина", "Показание"]
        if errs:
            headers += ["С идеальными приборами", "Методическая погрешность, %"]
        out.append({"caption": "Результаты измерений на постоянном токе", "headers": headers, "rows": rows})
    if "ac" in an:
        a = an["ac"]
        freqs = np.array(a["freq"])
        pick = np.unique(np.round(np.linspace(0, len(freqs) - 1, 9)).astype(int))
        names = list(a["readings"])
        scales = {n: col_scale([abs(v) for v in a["readings"][n]], meter_unit(circuit, n)) for n in names}
        headers = ["f, Гц"]
        for n in names:
            headers += [f"|{n}|, {scales[n][1]}", f"φ {n}, °"]
        rows = []
        for k in pick:
            row = [_num(freqs[k])]
            for n in names:
                v = a["readings"][n][k]
                row += [_num(abs(v) * scales[n][0]), config.dsep(f"{math.degrees(cmath.phase(v)):.1f}")]
            rows.append(row)
        out.append({"caption": "Частотные характеристики", "headers": headers, "rows": rows})
    if "tran" in an:
        t = an["tran"]
        time = np.array(t["time"])
        pick = np.unique(np.round(np.linspace(0, len(time) - 1, 11)).astype(int))
        names = list(t["readings"])
        tk, tu = col_scale(list(time), "с")
        scales = {n: col_scale(t["readings"][n], meter_unit(circuit, n)) for n in names}
        headers = [f"t, {tu}"] + [f"{n}, {scales[n][1]}" for n in names]
        rows = []
        for k in pick:
            rows.append([_num(time[k] * tk)] + [_num(clean(t["readings"][n][k]) * scales[n][0]) for n in names])
        out.append({"caption": "Переходный процесс", "headers": headers, "rows": rows})
    return out
