"""Графики для отчёта: АЧХ/ФЧХ и переходные процессы. Модель тут не участвует.
Напряжения и токи рисуются на разных осях, масштаб подбирается сам (В, мВ, мА, мкА)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from . import config  # noqa: E402
from .circuit import Circuit, fmt_si  # noqa: E402
from .simulate import col_scale, cutoff_frequency, meter_unit, time_constant  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.color": "0.6",
    "axes.axisbelow": True,
})

_COMMA = FuncFormatter(lambda v, _: config.dsep(f"{v:g}"))


def _groups(circuit: Circuit, names: list[str]) -> dict[str, list[str]]:
    g: dict[str, list[str]] = {}
    for n in names:
        g.setdefault(meter_unit(circuit, n), []).append(n)
    return g


def plot_ac(circuit: Circuit, res: dict, out: Path) -> Path:
    f = np.array(res["freq"])
    groups = _groups(circuit, list(res["readings"]))
    n_rows = len(groups) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(6.3, 2.6 * n_rows + 0.4), sharex=True)
    phase_ax = axes[-1]
    for ax, (unit, names) in zip(axes, groups.items()):
        allv = [abs(v) for n in names for v in res["readings"][n]]
        k, u = col_scale(allv, unit)
        for name in names:
            v = np.array(res["readings"][name])
            line, = ax.semilogx(f, np.abs(v) * k, lw=1.6, label=name)
            phase_ax.semilogx(f, np.degrees(np.angle(v)), lw=1.6, color=line.get_color(), label=name)
            fc = cutoff_frequency(res["freq"], res["readings"][name])
            if fc and unit == "В":
                ax.axvline(fc, color="0.3", ls="--", lw=1)
                ax.annotate(f"$f_{{ср}}$ ≈ {fmt_si(fc, 'Гц')}", (fc, np.abs(v).max() * 0.707 * k),
                            xytext=(8, 8), textcoords="offset points", fontsize=9)
        ax.set_ylabel(f"{'Напряжение' if unit == 'В' else 'Ток'}, {u}")
        ax.legend(loc="best", fontsize=9)
        ax.yaxis.set_major_formatter(_COMMA)
    phase_ax.set_ylabel("Фаза φ, град")
    phase_ax.set_xlabel("Частота f, Гц")
    phase_ax.legend(loc="best", fontsize=9)
    phase_ax.yaxis.set_major_formatter(_COMMA)
    fig.text(0.99, 0.005, "Амплитуда источника — 1 В (1 А)", ha="right", va="bottom", fontsize=8, color="0.4")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_tran(circuit: Circuit, res: dict, out: Path) -> Path:
    t = np.array(res["time"])
    tk, tu = col_scale(list(t), "с")
    groups = _groups(circuit, list(res["readings"]))
    n_rows = len(groups)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6.3, 3.0 * n_rows + 0.4), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (unit, names) in zip(axes, groups.items()):
        allv = [v for n in names for v in res["readings"][n]]
        k, u = col_scale(allv, unit)
        for name in names:
            v = np.array(res["readings"][name])
            ax.plot(t * tk, v * k, lw=1.6, label=name)
            tau = time_constant(res["time"], res["readings"][name])
            if tau:
                p = int(np.argmax(np.abs(v - v[-1])))
                y = (v[-1] + 0.368 * (v[p] - v[-1])) * k
                x = (t[p] + tau) * tk
                ax.plot([x], [y], "o", ms=4, color="0.2")
                ax.annotate(f"τ ≈ {fmt_si(tau, 'с')}", (x, y), xytext=(8, 4), textcoords="offset points", fontsize=9)
        ax.set_ylabel(f"{'Напряжение' if unit == 'В' else 'Ток'}, {u}")
        ax.legend(loc="best", fontsize=9)
        ax.yaxis.set_major_formatter(_COMMA)
        ax.xaxis.set_major_formatter(_COMMA)
    axes[-1].set_xlabel(f"Время t, {tu}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def make_plots(circuit: Circuit, sim: dict, out_dir: Path) -> list[dict]:
    """Возвращает [{'path', 'caption'}] для всех анализов, где нужен график."""
    figs = []
    an = sim["analyses"]
    if "ac" in an and an["ac"]["readings"]:
        p = plot_ac(circuit, an["ac"], out_dir / "plot_ac.png")
        figs.append({"path": str(p), "caption": "Амплитудно- и фазочастотные характеристики"})
    if "tran" in an and an["tran"]["readings"]:
        p = plot_tran(circuit, an["tran"], out_dir / "plot_tran.png")
        figs.append({"path": str(p), "caption": "Переходный процесс"})
    return figs
