"""Генерация SPICE-нетлиста и запуск ngspice в пакетном режиме.

Приборы моделируются честно, как в реальности:
    амперметр  = резистор r последовательно + источник 0 В (через него меряем ток)
    вольтметр  = резистор r параллельно участку (меряем разность потенциалов)
Идеальные приборы: амперметр без резистора, вольтметр вообще не подключается.

Каждый анализ запускается отдельным процессом ngspice, результаты пишутся
командой wrdata в текстовый файл и разбираются по именам векторов.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .circuit import GROUND, Analysis, Circuit, Element
from . import config


def available() -> bool:
    return shutil.which(config.NGSPICE_BIN) is not None


class _Names:
    """SPICE не любит кириллицу и произвольные имена, поэтому всё переименовываем."""

    def __init__(self, circuit: Circuit):
        self.node = {n: str(i + 1) for i, n in enumerate(circuit.nodes())}
        self.node[GROUND] = "0"
        self.elem: dict[str, str] = {}
        for k, e in enumerate(circuit.elements, start=1):
            letter = {"R": "R", "C": "C", "L": "L", "V": "V", "I": "I", "AM": "V", "VM": "R"}[e.type]
            self.elem[e.name] = f"{letter}x{k}"

    def v(self, node: str) -> str:
        return "0" if node == GROUND else f"v({self.node[node]})"


def _source_spec(e: Element, an_type: str, ac_mag: float) -> str:
    parts = []
    if an_type == "tran" and e.wave == "step":
        amp = e.amplitude if e.amplitude is not None else (e.value or 0.0)
        parts.append(f"PWL(0 0 1n {amp:g})")
    elif an_type == "tran" and e.wave == "sine":
        amp = e.amplitude if e.amplitude is not None else (e.value or 0.0)
        offset = e.value if (e.amplitude is not None and e.value) else 0.0
        parts.append(f"SIN({offset:g} {amp:g} {e.freq:g} 0 0 {e.phase:g})")
    else:
        parts.append(f"DC {(e.value or 0.0):g}")
    if an_type == "ac" and ac_mag:
        parts.append(f"AC {ac_mag:g}")
    return " ".join(parts)


def netlist(circuit: Circuit, an: Analysis) -> tuple[str, list[str], dict]:
    """Возвращает (текст нетлиста, список имён выходных векторов, карта вектор → прибор)."""
    nm = _Names(circuit)
    lines = [f"* {circuit.title}"]
    first_src = next((e.name for e in circuit.elements if e.is_source), None)
    any_ac = any(e.ac for e in circuit.elements)
    for e in circuit.elements:
        a, b = nm.node[e.n1], nm.node[e.n2]
        ref = nm.elem[e.name]
        if e.type in ("R", "C", "L"):
            lines.append(f"{ref} {a} {b} {e.value:g}")
        elif e.type in ("V", "I"):
            mag = (e.ac or 0.0) if any_ac else (1.0 if e.name == first_src else 0.0)
            lines.append(f"{ref} {a} {b} {_source_spec(e, an.type, mag)}")
        elif e.type == "AM":
            if e.r:
                mid = f"am{ref.lower()}"
                lines.append(f"R{ref[1:]}r {a} {mid} {e.r:g}")
                lines.append(f"{ref} {mid} {b} DC 0")
            else:
                lines.append(f"{ref} {a} {b} DC 0")
        elif e.type == "VM":
            if e.r:
                lines.append(f"{ref} {a} {b} {e.r:g}")

    # что сохраняем: показания приборов (+ потенциалы узлов для op)
    lets, outs, mapping = [], [], {}
    meters = circuit.meters()
    for k, m in enumerate(meters):
        if m.type == "VM":
            expr = f"{nm.v(m.n1)} - {nm.v(m.n2)}"
        else:
            expr = f"i({nm.elem[m.name]})"
        if an.type == "ac":
            lets.append(f"let m{k}re = real({expr})")
            lets.append(f"let m{k}im = imag({expr})")
            outs += [f"m{k}re", f"m{k}im"]
            mapping[f"m{k}"] = m.name
        else:
            lets.append(f"let m{k} = {expr}")
            outs.append(f"m{k}")
            mapping[f"m{k}"] = m.name
    if an.type == "op":
        for node, idx in nm.node.items():
            if node == GROUND:
                continue
            lets.append(f"let n{idx} = v({idx})")
            outs.append(f"n{idx}")
            mapping[f"n{idx}"] = f"node:{node}"
    if an.type == "ac":
        lets.insert(0, "let fr = real(frequency)")
        outs.insert(0, "fr")
        cmd = f"ac dec {an.points_per_decade} {an.f_start:g} {an.f_stop:g}"
    elif an.type == "tran":
        lets.insert(0, "let tt = time")
        outs.insert(0, "tt")
        tmax = min(an.t_step, an.t_stop / 200)
        cmd = f"tran {an.t_step:g} {an.t_stop:g} 0 {tmax:g} uic"
    else:
        cmd = "op"

    lines += [
        ".control",
        "set wr_vecnames",
        "set wr_singlescale",
        "set noaskquit",
        cmd,
        *lets,
        f"wrdata out.txt {' '.join(outs)}",
        "quit",
        ".endc",
        ".end",
    ]
    return "\n".join(lines) + "\n", outs, mapping


def _parse_wrdata(text: str, names: list[str]) -> dict[str, list[float]]:
    """Разбор вывода wrdata. Колонки берём с конца по нашим именам — так не важно,
    печатает ли ngspice ещё и колонку шкалы (времени или частоты)."""
    rows = [ln.split() for ln in text.strip().splitlines() if ln.strip()]
    if rows and not _is_number(rows[0][0]):
        rows = rows[1:]  # строка заголовков
    data = {n: [] for n in names}
    for row in rows:
        vals = row[-len(names):]
        for n, v in zip(names, vals):
            data[n].append(float(v))
    return data


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def run(circuit: Circuit, an: Analysis, keep_dir: Path | None = None) -> dict:
    if not available():
        raise RuntimeError("ngspice не найден (установите: sudo pacman -S ngspice)")
    text, outs, mapping = netlist(circuit, an)
    with tempfile.TemporaryDirectory(prefix="labspice_") as tmp:
        tmp = Path(tmp)
        (tmp / "circuit.cir").write_text(text, encoding="utf-8")
        proc = subprocess.run(
            [config.NGSPICE_BIN, "-b", "circuit.cir"],
            cwd=tmp, capture_output=True, text=True, timeout=120,
        )
        out_file = tmp / "out.txt"
        if keep_dir:
            keep_dir.mkdir(parents=True, exist_ok=True)
            (keep_dir / f"netlist_{an.type}.cir").write_text(text, encoding="utf-8")
            (keep_dir / f"ngspice_{an.type}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        if not out_file.exists():
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
            raise RuntimeError("ngspice не посчитал цепь:\n" + "\n".join(tail))
        data = _parse_wrdata(out_file.read_text(), outs)

    if an.type == "op":
        res = {"nodes": {}, "readings": {}}
        for key, target in mapping.items():
            val = data[key][-1]
            if target.startswith("node:"):
                res["nodes"][target[5:]] = val
            else:
                res["readings"][target] = val
        return res
    if an.type == "ac":
        res = {"freq": data["fr"], "readings": {}}
        for key, target in mapping.items():
            res["readings"][target] = [complex(r, i) for r, i in zip(data[f"{key}re"], data[f"{key}im"])]
        return res
    res = {"time": data["tt"], "readings": {}}
    for key, target in mapping.items():
        res["readings"][target] = data[key]
    return res
