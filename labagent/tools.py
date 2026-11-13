"""Инструменты агента. Одни и те же функции вызывает и MCP-сервер (режим «агент»),
и жёсткий пайплайн. Всё, что получилось, складывается в папку запуска runs/<id>/:

    circuit.json   описание цепи, которое составила модель
    sim.json       результаты симуляции
    schematic.png  схема
    plot_*.png     графики
    netlist_*.cir  нетлисты ngspice (если считал ngspice)
    report.docx / report.pdf
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
from pathlib import Path

from pydantic import ValidationError

from . import config
from .circuit import Circuit, load_circuit
from .circuit import dangling_problem
from .dc_lab import analyze as analyze_dc, structure_problems, target_mismatch
from .dc_lab import is_dc_lab
from .dc_lab import report_blocks as dc_report_blocks
from .dc_lab import text_summary as dc_text_summary
from .plots import make_plots
from .report import ReportData, build_docx, report_filename, to_pdf
from .schematic import draw_schematic
from .simulate import simulate, summary_text, tables

log = logging.getLogger(__name__)


def _jsonable(obj):
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _restore_sim(sim: dict) -> dict:
    ac = sim.get("analyses", {}).get("ac")
    if ac:
        ac["readings"] = {k: [complex(*v) for v in vals] for k, vals in ac["readings"].items()}
    return sim


def format_validation_error(exc: Exception) -> str:
    """Понятное модели описание ошибки — чтобы она могла исправиться сама."""
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            msg = err["msg"].replace("Value error, ", "")
            parts.append(f"{loc}: {msg}" if loc else msg)
        return "; ".join(parts)
    return str(exc)


class Toolbox:
    def __init__(self, llm=None, rag=None, runs_dir: Path | None = None, universal: bool = False,
                 task_text: str = ""):
        self.llm = llm
        # текст задания: по нему код проверяет, что прибор подключён туда, куда нужно
        self.task_text = task_text
        self._rejects = 0
        self._last_bad = None   # последняя отвергнутая цепь и сколько раз подряд её прислали
        self._repeat = 0
        self.rag = rag
        self.runs_dir = Path(runs_dir or config.RUNS_DIR)
        # универсальный режим: никаких готовых процедур лабораторных — агент считает сам
        self.universal = universal

    # ------------------------------------------------------------ RAG
    def search_docs(self, query: str, k: int = 4) -> str:
        if self.rag is None:
            return "ОШИБКА: база знаний не проиндексирована (python -m labagent index)."
        from .rag import format_context

        try:
            res = self.rag.search(query, self.llm, k=k)
        except Exception as exc:  # например, не запущен сервер эмбеддингов — ищем по словам
            log.warning("векторный поиск недоступен (%s), ищу по словам", exc)
            res = self.rag.search(query, None, k=k, mode="bm25")
        return format_context(res) if res else "Ничего не найдено."

    # ------------------------------------------------------------ симуляция
    def new_run(self, title: str) -> Path:
        slug = re.sub(r"[^\w]+", "-", title.lower(), flags=re.U).strip("-")[:30] or "run"
        run = self.runs_dir / f"{dt.datetime.now():%Y%m%d-%H%M%S}-{slug}"
        k = 1
        while run.exists():
            run = run.with_name(f"{run.name}-{k}")
            k += 1
        run.mkdir(parents=True)
        return run

    def simulate(self, circuit_data, force: bool = False) -> dict:
        """Внутренний вызов: проверить цепь, посчитать, нарисовать. Возвращает dict.
        force — считать, даже если у кода есть замечания к цепи (лучше криво, чем никак)."""
        key = circuit_data if isinstance(circuit_data, str) else repr(circuit_data)
        same = key == self._last_bad
        self._repeat = self._repeat + 1 if same else 0
        again = (" Ты прислал ту же самую цепь без изменений — так ошибка не исчезнет, измени описание."
                 if same else "")
        try:
            circuit = circuit_data if isinstance(circuit_data, Circuit) else load_circuit(circuit_data)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            self._last_bad = key
            return {"ok": False, "error": "описание цепи некорректно: " + format_validation_error(exc) + again}
        wrong = (dangling_problem(circuit)
                 or (target_mismatch(self.task_text, circuit) if self.task_text else None)
                 or structure_problems(circuit))
        # замечание возвращаем модели дважды; на третий раз (или на повтор той же цепи) считаем как есть
        if wrong and not force and self._rejects < 2 and self._repeat < 1:
            self._rejects += 1
            self._last_bad = key
            return {"ok": False, "error": "схема прочитана с ошибкой: " + wrong + again}
        run = self.new_run(circuit.title)
        (run / "circuit.json").write_text(circuit.model_dump_json(indent=2), encoding="utf-8")
        try:
            sim = simulate(circuit, keep_dir=run, with_ideal=not self.universal)
        except Exception as exc:
            return {"ok": False, "error": f"ошибка расчёта: {exc}", "run_id": run.name}
        (run / "sim.json").write_text(json.dumps(_jsonable(sim), ensure_ascii=False), encoding="utf-8")
        schematic = draw_schematic(circuit, run / "schematic.png", run / "schematic.svg")
        figures = make_plots(circuit, sim, run)
        (run / "figures.json").write_text(json.dumps(figures, ensure_ascii=False), encoding="utf-8")
        summary = summary_text(circuit, sim)
        if wrong:
            summary = f"ВНИМАНИЕ: у кода есть замечание к схеме, посчитано как есть: {wrong}\n{summary}"
        profile = "generic"
        if is_dc_lab(circuit) and not self.universal:
            # лабораторная «Цепи постоянного тока»: расчёт по Кирхгофу, r, δ, Δ0, Δ
            profile = "dc_lab"
            summary = f"{summary}\n{dc_text_summary(analyze_dc(circuit))}"
        (run / "profile.txt").write_text(profile, encoding="utf-8")
        return {"ok": True, "run_id": run.name, "run_dir": str(run), "circuit": circuit, "sim": sim,
                "summary": summary, "schematic": str(schematic), "figures": figures, "profile": profile}

    def simulate_circuit(self, circuit) -> str:
        """Вариант для модели: короткий текст."""
        r = self.simulate(circuit)
        if not r["ok"]:
            return f"ОШИБКА: {r['error']}. Исправь описание цепи и вызови инструмент снова."
        files = [r["schematic"]] + [f["path"] for f in r["figures"]]
        return (f"run_id: {r['run_id']}\n{r['summary']}\n"
                f"Схема и графики построены автоматически: {', '.join(Path(f).name for f in files)}.\n"
                + (f"Данные для расчётов в run_python: /data/circuit.json и /data/sim.json (run_id={r['run_id']})."
                   if self.universal else f"Для отчёта вызови build_report с run_id={r['run_id']}."))

    def draw_schematic(self, circuit) -> str:
        try:
            c = load_circuit(circuit)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            return "ОШИБКА: " + format_validation_error(exc)
        run = self.new_run(c.title + "-schema")
        path = draw_schematic(c, run / "schematic.png", run / "schematic.svg")
        return f"Схема сохранена: {path}"

    # ------------------------------------------------------------ отчёт
    def build_report(self, run_id: str, lab_number: str, lab_title: str, goal: str = "",
                     conclusions: str = "", theory: str = "", procedure: str = "", variant: str = "") -> dict:
        run = self.runs_dir / run_id
        if not (run / "sim.json").exists():
            return {"ok": False, "error": f"нет результатов для run_id={run_id}: сначала simulate_circuit"}
        circuit = Circuit.model_validate_json((run / "circuit.json").read_text(encoding="utf-8"))
        sim = _restore_sim(json.loads((run / "sim.json").read_text(encoding="utf-8")))
        figures = json.loads((run / "figures.json").read_text(encoding="utf-8"))
        tbls = tables(circuit, sim)
        engine = "ngspice" if sim["engine"] == "ngspice" else "встроенный решатель (метод узловых потенциалов)"
        title = lab_title.strip().strip("«»\"")
        variant = str(variant or "").strip().rstrip(".")

        if is_dc_lab(circuit):
            res = analyze_dc(circuit)
            data = ReportData(lab_number=str(lab_number), lab_title=title, variant=variant, blocks_only=True,
                              blocks=dc_report_blocks(res, run, engine, conclusions.strip() or None))
            docx = build_docx(data, run / report_filename(lab_number))
            pdf = to_pdf(docx)
            return {"ok": True, "docx": str(docx), "pdf": str(pdf) if pdf else None, "run_id": run_id}

        meters = [e for e in circuit.meters()]
        meter_txt = "; ".join(
            f"{m.name} — {'вольтметр' if m.type == 'VM' else 'амперметр'}"
            + (f" с {'входным' if m.type == 'VM' else 'внутренним'} сопротивлением {m.r:g} Ом" if m.r else " (идеальный)")
            for m in meters)
        equipment = (f"Работа выполнена методом компьютерного моделирования, расчёт — {engine}. "
                     f"Измерительные приборы: {meter_txt or 'не используются'}.")

        n_fig = 1 + len(figures)
        blocks = [("heading", "Схема исследуемой цепи"),
                  ("text", "Схема исследуемой цепи приведена на рисунке 1."),
                  ("figure", str(run / "schematic.png"), f"Схема исследуемой цепи: {circuit.title}"),
                  ("heading", "Результаты измерений")]
        refs = []
        if tbls:
            refs.append("в таблице 1" if len(tbls) == 1 else f"в таблицах 1–{len(tbls)}")
        if figures:
            refs.append("на рисунке 2" if n_fig == 2 else f"на рисунках 2–{n_fig}")
        if refs:
            blocks.append(("text", "Результаты моделирования приведены " + " и ".join(refs) + "."))
        for t in tbls:
            blocks.append(("table", t))
        for f in figures:
            blocks.append(("figure", f["path"], f["caption"]))

        data = ReportData(lab_number=str(lab_number), lab_title=title, goal=goal, variant=variant,
                          theory=theory, equipment=equipment, procedure=procedure,
                          conclusions=conclusions, blocks=blocks)
        docx = build_docx(data, run / report_filename(lab_number))
        pdf = to_pdf(docx)
        return {"ok": True, "docx": str(docx), "pdf": str(pdf) if pdf else None, "run_id": run_id}

    def build_report_text(self, **kw) -> str:
        r = self.build_report(**kw)
        if not r["ok"]:
            return "ОШИБКА: " + r["error"]
        out = f"REPORT_DOCX: {r['docx']}"
        out += f"\nREPORT_PDF: {r['pdf']}" if r["pdf"] else "\n(PDF не создан: LibreOffice не найден)"
        return out + "\nОтчёт готов."

    # ------------------------------------------------------------ универсальный режим
    def _run_dir(self, run_id: str) -> Path | None:
        if not run_id or "/" in run_id or ".." in run_id:
            return None
        run = self.runs_dir / run_id
        return run if run.is_dir() else None

    def run_python(self, code: str, run_id: str = "") -> str:
        """Выполнить код модели в песочнице. /data — результаты simulate_circuit этого run_id."""
        from .sandbox import SandboxError, run_code

        run = self._run_dir(run_id)
        if run_id and run is None:
            return f"ОШИБКА: нет запуска run_id={run_id}"
        work = (run / "work") if run else self.new_run("scratch") / "work"
        try:
            r = run_code(code, work, data=run)
        except SandboxError as exc:
            return f"ОШИБКА песочницы: {exc}"
        out = r["stdout"][-4000:]
        err = r["stderr"][-1500:]
        text = f"код выхода: {r['returncode']} (песочница: {r['backend']}, {r['seconds']} с)\n"
        text += f"stdout:\n{out or '(пусто)'}\n"
        if err.strip():
            text += f"stderr:\n{err}\n"
        if r["files"]:
            text += "созданы файлы в /work: " + ", ".join(r["files"])
        if r["returncode"] != 0:
            text += "\nКод завершился с ошибкой — исправь и запусти снова."
        return text

    def build_custom_report(self, run_id: str, lab_number: str, lab_title: str, sections: list,
                            variant: str = "", key_results: dict | None = None) -> dict:
        """Отчёт, структуру которого придумал сам агент: список разделов
        {"title", "text", "image"?, "table"? {"headers", "rows"}}."""
        run = self._run_dir(run_id)
        if run is None:
            return {"ok": False, "error": f"нет запуска run_id={run_id}: сначала simulate_circuit"}
        if isinstance(sections, str):
            sections = json.loads(sections)
        blocks, problems = [], []
        for k, sec in enumerate(sections or [], 1):
            if not isinstance(sec, dict):
                problems.append(f"раздел {k} — не объект")
                continue
            if sec.get("title"):
                blocks.append(("bold", str(sec["title"])))
            if sec.get("text"):
                blocks.append(("text", str(sec["text"])))
            img = sec.get("image")
            if img:
                name = Path(str(img)).name  # только имя файла — никаких путей наружу
                path = next((p for p in (run / "work" / name, run / name) if p.is_file()), None)
                if path is None:
                    problems.append(f"картинка {name} не найдена (сохраняй графики в /work)")
                else:
                    blocks.append(("figure", str(path), sec.get("title", "")))
            tbl = sec.get("table")
            if isinstance(tbl, dict) and tbl.get("headers") and tbl.get("rows"):
                blocks.append(("table", {"caption": sec.get("title", "Таблица"),
                                         "headers": [str(h) for h in tbl["headers"]],
                                         "rows": [[str(c) for c in row] for row in tbl["rows"]]}))
        if not blocks:
            return {"ok": False, "error": "в отчёте нет ни одного раздела"}
        if key_results:
            if isinstance(key_results, str):
                key_results = json.loads(key_results)
            (run / "key_results.json").write_text(json.dumps(key_results, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
        data = ReportData(lab_number=str(lab_number), lab_title=lab_title.strip().strip("«»\""),
                          variant=str(variant or "").rstrip("."), blocks=blocks, blocks_only=True)
        docx = build_docx(data, run / report_filename(lab_number))
        pdf = to_pdf(docx)
        # сразу проверяем кодом по списку замечаний — агент увидит их и сможет пересобрать отчёт
        from .review import auto_checks, read_report

        problems += [r for r in auto_checks(read_report(docx), docx) if "титульного листа" not in r and "Имя файла" not in r]
        return {"ok": True, "docx": str(docx), "pdf": str(pdf) if pdf else None, "run_id": run_id,
                "problems": problems}

    def build_custom_report_text(self, **kw) -> str:
        try:
            r = self.build_custom_report(**kw)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return f"ОШИБКА: {exc}"
        if not r["ok"]:
            return "ОШИБКА: " + r["error"]
        out = f"REPORT_DOCX: {r['docx']}" + (f"\nREPORT_PDF: {r['pdf']}" if r["pdf"] else "")
        if r["problems"]:
            out += ("\nЗамечания (исправь и вызови build_custom_report ещё раз): "
                    + "; ".join(r["problems"]))
        return out + "\nОтчёт готов."
