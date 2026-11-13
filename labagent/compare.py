"""Сравнение режимов «пайплайн» и «агент» на наборе заданий из папки tasks/.

Каждое задание содержит эталонную цепь. Результат засчитывается, если:
  * отчёт создан;
  * показания приборов совпадают с эталоном (постоянный ток — 1 %),
    частота среза — 3 %, постоянная времени — 3 %.
Так проверяется не «красивость» ответа, а правильность цепи, которую собрала модель.
"""
from __future__ import annotations

import datetime as dt
import json
import traceback
from pathlib import Path

from . import config
from .circuit import Circuit, load_circuit
from .simulate import cutoff_frequency, simulate, time_constant
from .tools import _restore_sim


def _features(circuit: Circuit, sim: dict) -> dict:
    f = {"op": {}, "fc": [], "tau": []}
    an = sim["analyses"]
    if "op" in an:
        for m in circuit.meters():
            f["op"].setdefault(m.type, []).append(an["op"]["readings"][m.name])
    if "ac" in an:
        for name, vals in an["ac"]["readings"].items():
            fc = cutoff_frequency(an["ac"]["freq"], vals)
            if fc:
                f["fc"].append(fc)
    if "tran" in an:
        for name, vals in an["tran"]["readings"].items():
            tau = time_constant(an["tran"]["time"], vals)
            if tau:
                f["tau"].append(tau)
    return f


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-15)


def score(reference: Circuit, run_dir: Path) -> tuple[bool, str]:
    if not (run_dir / "sim.json").exists():
        return False, "нет результатов симуляции"
    got_c = Circuit.model_validate_json((run_dir / "circuit.json").read_text(encoding="utf-8"))
    got = _features(got_c, _restore_sim(json.loads((run_dir / "sim.json").read_text(encoding="utf-8"))))
    ref = _features(reference, simulate(reference, engine="builtin"))
    problems = []
    for mtype, vals in ref["op"].items():
        for v in vals:
            # полярность прибора на картинке задания не указана — знак показания не сравниваем
            if not any(_close(abs(v), abs(g), 0.01) for g in got["op"].get(mtype, [])):
                problems.append(f"нет показания {mtype} ≈ {v:.4g}")
    for key, tol, label in (("fc", 0.03, "частота среза"), ("tau", 0.03, "τ")):
        for v in ref[key]:
            if not any(_close(v, g, tol) for g in got[key]):
                problems.append(f"{label} ≈ {v:.4g} не найдена (есть: {[round(g, 4) for g in got[key]]})")
    return not problems, "; ".join(problems) or "совпадает с эталоном"


def score_numbers(reference: Circuit, run_dir: Path) -> tuple[bool, str]:
    """Для универсального агента: совпадают ли его U0 (I0), r и Δ0 с эталонным расчётом."""
    from .dc_lab import analyze, is_dc_lab

    path = run_dir / "key_results.json"
    if not is_dc_lab(reference):
        ref_f = _features(reference, simulate(reference, engine="builtin"))
        if not ref_f["tau"] and not ref_f["fc"]:
            return True, "не проверяются (нет эталонной величины)"
        if not path.exists():
            return False, "агент не передал key_results"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            got = {str(k).lower(): float(str(v).replace(",", ".")) for k, v in raw.items()
                   if isinstance(v, (int, float, str)) and str(v).replace(",", ".").replace(".", "", 1).replace("e-", "", 1).isdigit()}
        except Exception as exc:
            return False, f"key_results не читается: {exc}"
        bad = []
        for key, vals in (("tau", ref_f["tau"]), ("f_c", ref_f["fc"])):
            if not vals:
                continue
            cand = [v for k, v in got.items() if k.startswith(key.split("_")[0])]
            if not any(_close(vals[0], g, 0.03) for g in cand):
                bad.append(f"{key} ≈ {vals[0]:.4g} не найдена (есть: {[round(c, 5) for c in cand]})")
        return not bad, "; ".join(bad) or "главные величины совпадают с эталоном"

    if not path.exists():
        return False, "агент не передал key_results"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        got = {}
        for k, v in raw.items():
            try:
                got[str(k).lower()] = float(str(v).replace(",", ".").replace("−", "-"))
            except ValueError:
                pass
    except Exception as exc:
        return False, f"key_results не читается: {exc}"
    ref = analyze(reference)
    checks = [("x0", ref.x0, 0.01), ("r", ref.r, 0.02), ("delta0", ref.delta0, 0.05)]
    bad = []
    for key, val, tol in checks:
        if key not in got:
            bad.append(f"нет {key}")
        elif not (abs(got[key] - val) <= tol * max(abs(val), 1e-12) or abs(got[key] - val) < 1e-9):
            bad.append(f"{key} = {got[key]:.4g} вместо {val:.4g}")
    return not bad, "; ".join(bad) or "U0, r, Δ0 совпадают с эталоном"


def run_compare(modes: list[str], repeats: int, task_files: list[Path], log=print,
                text_only: bool = False) -> Path:
    """text_only — не передавать картинки (если модель их не понимает)."""
    args_text_only = text_only
    from .agent import run_agent
    from .llm import LLM
    from .pipeline import run_pipeline
    from .inputs import prepare, spec_files
    from .manual import load_rag
    from .review import review_report
    from .tools import Toolbox

    rows = []
    for tf in task_files:
        spec = json.loads(tf.read_text(encoding="utf-8"))
        reference = load_circuit(spec["reference"])
        files, manuals = spec_files(spec, tf.parent)
        # задание и методичка — свои у каждой работы; постоянная база (ГОСТ, замечания) — общая
        task_text, images, task_index = prepare(spec.get("task", ""), files, manuals, args_text_only, LLM(), log,
                                                variant=spec.get("variant"))
        rag = load_rag(task_index)
        for mode in modes:
            for k in range(1, repeats + 1):
                log(f"\n=== {tf.stem} | {mode} | прогон {k}/{repeats} ===")
                llm = LLM()
                try:
                    if mode == "pipeline":
                        res = run_pipeline(task_text, llm=llm, rag=rag, toolbox=Toolbox(llm=llm, rag=rag),
                                           log=log, images=images)
                    else:
                        res = run_agent(task_text, llm=llm, log=log, images=images, task_index=task_index,
                                        universal=(mode == "universal"), max_steps=24 if mode == "universal" else 14)
                except Exception as exc:
                    res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    log(traceback.format_exc())
                run_dir = Path(res["report_docx"]).parent if res.get("report_docx") else None
                if run_dir is None and res.get("run_ids"):
                    run_dir = config.RUNS_DIR / res["run_ids"][-1]
                correct, note = score(reference, run_dir) if run_dir else (False, res.get("error", "нет запуска"))
                if mode == "universal":
                    nums_ok, nums_note = score_numbers(reference, run_dir) if run_dir else (False, "нет отчёта")
                else:
                    # в специальных режимах числа считает проверенный код: верны, если верна цепь
                    nums_ok, nums_note = correct, "считает код"
                note = f"{note}; числа: {nums_note}"
                remarks = None
                if res.get("report_docx"):  # автоматические проверки по списку замечаний (без модели — быстро)
                    try:
                        remarks = len(review_report(res["report_docx"], llm=None)["auto"])
                    except Exception as exc:
                        log(f"проверка отчёта не удалась: {exc}")
                rows.append({"task": tf.stem, "mode": mode, "run": k, "report": bool(res.get("report_docx")),
                             "correct": correct, "numbers": nums_ok, "note": note, "llm_calls": res.get("llm_calls"),
                             "seconds": res.get("seconds"), "steps": res.get("steps"),
                             "fix_attempts": res.get("fix_attempts"), "remarks": remarks})
                log(f"→ отчёт: {'да' if rows[-1]['report'] else 'нет'}, цепь верна: {'да' if correct else 'нет'}, "
                    f"числа верны: {'да' if nums_ok else 'нет'} ({note})")

    out_dir = config.ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = f"{dt.datetime.now():%Y%m%d-%H%M%S}"
    (out_dir / f"compare-{stamp}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# Сравнение режимов ({config.LLM_MODEL})", "",
             "| Задание | Режим | Отчёт создан | Цепь верна | Числа верны | Замечаний (авто) | Среднее время, с | Вызовов LLM |",
             "|---|---|---|---|---|---|---|---|"]
    for tf in task_files:
        for mode in modes:
            rs = [r for r in rows if r["task"] == tf.stem and r["mode"] == mode]
            if not rs:
                continue
            t = [r["seconds"] for r in rs if r["seconds"]]
            c = [r["llm_calls"] for r in rs if r["llm_calls"]]
            rm = [r["remarks"] for r in rs if r.get("remarks") is not None]
            lines.append(f"| {tf.stem} | {mode} | {sum(r['report'] for r in rs)}/{len(rs)} | "
                         f"{sum(r['correct'] for r in rs)}/{len(rs)} | {sum(r['numbers'] for r in rs)}/{len(rs)} | "
                         + (f"{sum(rm) / len(rm):.1f} | " if rm else "— | ")
                         + (f"{sum(t) / len(t):.0f} | {sum(c) / len(c):.1f} |" if t and c else "— | — |"))
    lines += ["", "## Подробно", ""]
    for r in rows:
        lines.append(f"- {r['task']} / {r['mode']} / #{r['run']}: отчёт {'✓' if r['report'] else '✗'}, "
                     f"цепь {'✓' if r['correct'] else '✗'}, числа {'✓' if r['numbers'] else '✗'} — {r['note']}")
    md = out_dir / f"compare-{stamp}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n" + "\n".join(lines[:4 + len(task_files) * len(modes)]))
    return md
