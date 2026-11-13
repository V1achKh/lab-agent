"""Режим «пайплайн»: шаги жёстко заданы в коде, модель вызывается в одном-двух местах.

    1. RAG: найти методичку                                      (код)
    2. LLM: задание (текст и/или картинка) → описание цепи (JSON)  ← модель
       ↳ если JSON неверный — вернуть ошибку модели, до 3 попыток
    3. Симуляция, схема, графики, таблицы                         (код)
       для лабораторной №1 — ещё и уравнения Кирхгофа, r, δ, Δ0, Δ (код)
    4. LLM: цель, теория, выводы — только для «прочих» работ       ← модель
       (в лабораторной №1 вывод формируется из чисел — надёжнее)
    5. Отчёт, PDF                                                  (код)
"""
from __future__ import annotations

import json
import time

from pydantic import ValidationError

from .circuit import load_circuit
from .circuit import dangling_problem
from .dc_lab import structure_problems, target_mismatch
from .llm import LLM, extract_json, user_message
from .prompts import PIPELINE_CIRCUIT_SYSTEM, PIPELINE_TEXT_SYSTEM
from .rag import RagIndex, format_context
from .review import with_checklist
from .tools import Toolbox, format_validation_error


def _context(rag: RagIndex | None, llm: LLM, task: str) -> str:
    if rag is None:
        return ""
    seen, results = set(), []
    for query, k in ((task, 4), ("программа выполнения работы, содержание отчёта", 2)):
        try:
            found = rag.search(query, llm, k=k)
        except Exception:
            found = rag.search(query, None, k=k, mode="bm25")
        for r in found:
            key = (r["source"], r["text"][:80])
            if key not in seen:
                seen.add(key)
                results.append(r)
    return format_context(results)


def run_pipeline(task: str, llm: LLM | None = None, rag: RagIndex | None = None,
                 toolbox: Toolbox | None = None, max_fix: int = 3, log=print, images=None) -> dict:
    llm = llm or LLM()
    toolbox = toolbox or Toolbox(llm=llm, rag=rag)
    t0 = time.time()
    out = {"mode": "pipeline", "ok": False, "fix_attempts": 0}

    # 1. RAG
    ctx = _context(rag, llm, task)
    log(f"[1/5] RAG: найдено фрагментов — {ctx.count(chr(10) + '[') + (1 if ctx else 0)}")

    # 2. описание цепи (с самоисправлением)
    text = f"Фрагменты документов:\n{ctx or '(нет)'}\n\nЗадание:\n{task}"
    messages = [{"role": "system", "content": PIPELINE_CIRCUIT_SYSTEM}, user_message(text, images)]
    spec, sim = {}, None
    soft = None          # цепь, которая считается, но с замечаниями кода (берётся, если лучше не будет)
    raw_last = None      # последнее описание модели, даже некорректное (для автоисправления)
    for attempt in range(1, max_fix + 1):
        reply = llm.chat(messages, json_mode=True)
        messages.append({"role": "assistant", "content": reply["content"]})
        try:
            spec = extract_json(reply["content"])
            raw_last = spec.get("circuit", spec)
            circuit = load_circuit(raw_last)
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            err = format_validation_error(exc)
            log(f"[2/5] попытка {attempt}: ошибка в описании цепи — {err}")
            out["fix_attempts"] += 1
            messages.append({"role": "user", "content": f"Ошибка: {err}\nИсправь и пришли JSON целиком."})
            continue
        # прибор там, где требует задание, и включён правильно? (проверка кодом — мягкая:
        # модель получает замечание и пробует исправить, но если не смогла — работаем с тем, что есть)
        wrong = dangling_problem(circuit) or target_mismatch(task, circuit) or structure_problems(circuit)
        if wrong:
            log(f"[2/5] попытка {attempt}: {wrong}")
            soft = (circuit, wrong)
            if attempt < max_fix:
                out["fix_attempts"] += 1
                messages.append({"role": "user", "content": f"Ошибка: {wrong}\nИсправь и пришли JSON целиком."})
                continue
            break
        soft = None
        sim = toolbox.simulate(circuit)
        if sim["ok"]:
            log(f"[2/5] цепь «{circuit.title}»: {len(circuit.elements)} элементов (попытка {attempt})")
            break
        log(f"[2/5] попытка {attempt}: {sim['error']}")
        out["fix_attempts"] += 1
        messages.append({"role": "user", "content": f"Ошибка: {sim['error']}\nИсправь и пришли JSON целиком."})
        sim = None
    warnings = []
    if sim is None and soft is not None:
        circuit, wrong = soft
        warnings.append(f"модель не исправила замечание, отчёт по её цепи как есть: {wrong}")
        sim = toolbox.simulate(circuit, force=True)
    if sim is None or not sim.get("ok"):
        out.update(error="модель не смогла составить корректную цепь (каждая попытка — с ошибкой, см. выше)",
                   seconds=round(time.time() - t0, 1),
                   llm_calls=llm.calls)
        return out
    for w in warnings:
        log(f"[2/5] ВНИМАНИЕ: {w}")
    out["warnings"] = warnings

    # 3. расчёт уже выполнен в toolbox.simulate
    log(f"[3/5] расчёт ({sim['sim']['engine']}), схема готова: {sim['run_id']}")
    log("      " + sim["summary"].replace("\n", "\n      "))
    log(f"      Сравните схему с заданием: {sim['schematic']}")

    # 4. тексты отчёта (для лабораторной №1 не нужны — всё считается)
    texts = {}
    if sim["profile"] != "dc_lab":
        text_user = (f"Фрагменты методички:\n{ctx or '(нет)'}\n\nЗадание: {task}\n\n"
                     f"Результаты моделирования:\n{sim['summary']}")
        for _ in range(2):
            try:
                texts = extract_json(llm.ask(with_checklist(PIPELINE_TEXT_SYSTEM), text_user, json_mode=True))
                break
            except ValueError:
                continue
        log(f"[4/5] тексты отчёта: {', '.join(k for k in ('goal', 'theory', 'conclusions') if texts.get(k))}")
    else:
        log("[4/5] лабораторная №1: уравнения, погрешности и вывод сформированы по результатам расчёта")

    # 5. отчёт
    rep = toolbox.build_report(
        run_id=sim["run_id"], lab_number=str(spec.get("lab_number") or "—"),
        lab_title=spec.get("lab_title") or sim["circuit"].title, variant=str(spec.get("variant") or ""),
        goal=texts.get("goal", ""), theory=texts.get("theory", ""), conclusions=texts.get("conclusions", ""))
    log(f"[5/5] отчёт: {rep.get('docx')}" + (f" и {rep['pdf']}" if rep.get("pdf") else ""))
    out.update(ok=rep["ok"], run_id=sim["run_id"], report_docx=rep.get("docx"), report_pdf=rep.get("pdf"),
               circuit=sim["circuit"].model_dump(), summary=sim["summary"], llm_calls=llm.calls,
               seconds=round(time.time() - t0, 1), tokens=llm.tokens)
    return out
