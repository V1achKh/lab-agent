"""Режим «агент»: модель сама решает, какие инструменты вызывать.

Схема работы:
    агент (этот файл) ── stdio ──► MCP-сервер (mcp_server.py) ──► tools.py
          │
          └── HTTP ──► локальная модель (Ollama / llama-server)

Цикл: отправляем модели историю + список инструментов → если она просит вызвать
инструмент, вызываем его через MCP и добавляем результат в историю → повторяем,
пока модель не ответит обычным текстом (или не кончится лимит шагов).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from . import config
from .llm import LLM, user_message
from .prompts import AGENT_SYSTEM, AGENT_UNIVERSAL_SYSTEM
from .review import with_checklist


def _schema(tool) -> dict:
    # mcp 2.x: input_schema, mcp 1.x: inputSchema
    return getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {"type": "object"}


def _result_text(result) -> str:
    parts = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    text = "\n".join(parts)
    if getattr(result, "is_error", None) or getattr(result, "isError", None):
        text = "ОШИБКА: " + text
    return text


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


async def run_agent_async(task: str, llm: LLM | None = None, max_steps: int = 14, log=print,
                          images=None, universal: bool = False, task_index=None) -> dict:
    """images — картинки задания; task_index — индекс методички этого задания (см. manual.py):
    передаётся MCP-серверу, и search_docs ищет в постоянной базе и в методичке сразу."""
    llm = llm or LLM()
    env = {**os.environ, "LAB_AGENT_MODE": "universal" if universal else "special"}
    env.pop("LAB_TASK_INDEX", None)
    # текст задания (без методички) — MCP-сервер по нему проверяет, что прибор подключён куда нужно
    env["LAB_TASK_TEXT"] = task.split("Методические указания к этой работе")[0][:4000]
    if task_index:
        env["LAB_TASK_INDEX"] = str(task_index)
    params = StdioServerParameters(command=sys.executable, args=["-m", "labagent.mcp_server"],
                                   cwd=str(config.ROOT), env=env)
    t0 = time.time()
    trace = []
    final = ""
    report = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            tools = [{"type": "function", "function": {
                "name": t.name, "description": t.description or "", "parameters": _schema(t)}}
                for t in mcp_tools]
            log(f"MCP: подключено инструментов — {len(tools)}: {', '.join(t.name for t in mcp_tools)}")
            system = with_checklist(AGENT_UNIVERSAL_SYSTEM if universal else AGENT_SYSTEM)
            messages = [{"role": "system", "content": system},
                        user_message(task, images)]
            error = None
            seen: dict[str, int] = {}   # одинаковый вызов с одинаковой ошибкой — признак зацикливания
            for step in range(1, max_steps + 1):
                if error:
                    break
                try:
                    msg = await asyncio.to_thread(llm.chat, messages, tools)
                except Exception as exc:  # модель недоступна, оборвалось соединение и т.п.
                    error = f"ошибка обращения к модели: {exc}"
                    log(error)
                    break
                messages.append(msg)
                calls = msg.get("tool_calls") or []
                if not calls:
                    final = msg.get("content", "")
                    break
                for call in calls:
                    name = call["function"]["name"]
                    args = _parse_args(call["function"].get("arguments"))
                    short = json.dumps(args, ensure_ascii=False)
                    log(f"[шаг {step}] → {name}({short[:160]}{'…' if len(short) > 160 else ''})")
                    try:
                        result = await session.call_tool(name, args)
                        text = _result_text(result)
                    except Exception as exc:  # неверные аргументы и т.п. — модель увидит и исправит
                        text = f"ОШИБКА вызова инструмента: {exc}"
                    log(f"          ← {text[:300].replace(chr(10), ' | ')}")
                    if text.startswith("ОШИБКА"):
                        sig = name + json.dumps(args, ensure_ascii=False, sort_keys=True) + text[:200]
                        seen[sig] = seen.get(sig, 0) + 1
                        if seen[sig] >= 3:
                            error = (f"модель зациклилась: {seen[sig]} раза подряд вызвала {name} с теми же "
                                     "данными и получила ту же ошибку — остановлено")
                            log(error)
                    trace.append({"step": step, "tool": name, "args": args, "result": text})
                    for key in ("REPORT_DOCX", "REPORT_PDF"):
                        m = re.search(rf"{key}: (.+)", text)
                        if m:
                            report[key.lower()] = m.group(1).strip()
                    messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                                     "name": name, "content": text})
            else:
                final = "(лимит шагов исчерпан)"
    fallback = None
    if not report.get("report_docx"):
        # отчёт обязан быть: агент не довёл дело до конца — код досчитывает его цепь «как есть»
        fallback = _fallback_report(task, trace, llm, images, task_index, log)
        if fallback:
            report = {"report_docx": fallback.get("docx"), "report_pdf": fallback.get("pdf")}
    return {
        "error": error, "fallback": bool(fallback),
        "mode": "universal" if universal else "agent", "ok": bool(report.get("report_docx")), "answer": final,
        "report_docx": report.get("report_docx"), "report_pdf": report.get("report_pdf"),
        "steps": len(trace), "trace": trace, "llm_calls": llm.calls,
        "seconds": round(time.time() - t0, 1), "tokens": llm.tokens,
        "run_ids": sorted({m.group(1) for t in trace for m in [re.search(r"run_id: (\S+)", t["result"])] if m}),
    }


def _lab_meta(task: str) -> tuple[str, str, str]:
    num = re.search(r"(?:лабораторн\w*\s+работ\w*|лаб\.?)\s*№?\s*(\d+)", task, re.I)
    title = re.search(r"[«\"]([^»\"]{5,80})[»\"]", task)
    var = re.search(r"вариант\w*\s*№?\s*([\wА-Яа-я]+)", task, re.I)
    return (num.group(1) if num else "1", title.group(1) if title else "Цепи постоянного тока",
            var.group(1) if var else "")


def _fallback_report(task, trace, llm, images, task_index, log) -> dict | None:
    """Запасной путь: собрать отчёт по последней цепи, которую прислала модель (даже с ошибкой).
    Если модель так и не прислала ни одной разборчивой цепи — отчёт делает pipeline."""
    from .tools import Toolbox

    log("Агент не собрал отчёт — собираю его сам по последней цепи модели (как есть, с замечаниями).")
    num, title, variant = _lab_meta(task)
    tb = Toolbox(llm=llm, task_text=task.split("Методические указания к этой работе")[0][:4000])
    run_id = None
    for t in reversed(trace):           # уже посчитанная цепь — лучше всего
        m = re.search(r"run_id: (\S+)", t["result"])
        if t["tool"] == "simulate_circuit" and m:
            run_id = m.group(1)
            break
    if run_id is None:
        for t in reversed(trace):       # иначе — последняя присланная цепь, считаем принудительно
            c = (t.get("args") or {}).get("circuit") or (t.get("args") or {}).get("circuit_json")
            if t["tool"] != "simulate_circuit" or not c:
                continue
            try:
                r = tb.simulate(c, force=True)
            except Exception as exc:
                log(f"  цепь не считается: {exc}")
                continue
            if r.get("ok"):
                run_id = r["run_id"]
                log("  " + r["summary"].replace("\n", "\n  "))
                break
    if run_id is None:
        log("  разборчивой цепи нет — запускаю pipeline (шаги задаёт код)")
        from .manual import load_rag
        from .pipeline import run_pipeline
        res = run_pipeline(task, llm=llm, rag=load_rag(task_index), images=images, log=log)
        return {"docx": res.get("report_docx"), "pdf": res.get("report_pdf")} if res.get("ok") else None
    try:
        rep = tb.build_report(run_id=run_id, lab_number=num, lab_title=title, variant=variant)
    except Exception as exc:
        log(f"  отчёт не собрался: {exc}")
        return None
    log(f"  отчёт: {rep.get('docx')}")
    return rep if rep.get("ok") else None


def run_agent(task: str, llm: LLM | None = None, max_steps: int = 14, log=print, images=None,
              universal: bool = False, task_index=None) -> dict:
    return asyncio.run(run_agent_async(task, llm, max_steps, log, images, universal, task_index))
