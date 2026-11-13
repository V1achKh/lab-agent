"""MCP-сервер с инструментами для лабораторных по электротехнике.

Запуск (stdio):  python -m labagent.mcp_server
Его можно подключить к любому MCP-клиенту, не только к нашему агенту.
Пример конфигурации клиента — в README.

Два набора инструментов (переменная окружения LAB_AGENT_MODE):
  special   (по умолчанию) — search_docs, simulate_circuit, draw_schematic, build_report;
            для лабораторной №1 вся процедура методички выполняется готовым кодом;
  universal — search_docs, simulate_circuit, draw_schematic, run_python, build_custom_report;
            готовых процедур нет: агент сам пишет код расчёта (он выполняется в песочнице)
            и сам решает, из каких разделов состоит отчёт.

Важно: в stdio-режиме stdout занят протоколом, поэтому всё логирование — в stderr.
"""
from __future__ import annotations

import logging
import os
import sys

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import config
from .llm import LLM
from .prompts import CIRCUIT_FORMAT
from .tools import Toolbox

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[mcp] %(message)s")
log = logging.getLogger("labagent.mcp")

MODE = os.environ.get("LAB_AGENT_MODE", "special")
UNIVERSAL = MODE == "universal"

server = _Server(
    "lab-agent",
    instructions="Инструменты для моделирования электрических цепей и оформления отчётов.",
)

_toolbox: Toolbox | None = None


def toolbox() -> Toolbox:
    global _toolbox
    if _toolbox is None:
        from .manual import load_rag

        llm = LLM()
        rag = load_rag()   # постоянная база + методичка задания (LAB_TASK_INDEX), если агент её передал
        _toolbox = Toolbox(llm=llm, rag=rag, runs_dir=config.RUNS_DIR, universal=UNIVERSAL,
                           task_text=os.environ.get("LAB_TASK_TEXT", ""))
    return _toolbox


@server.tool()
def search_docs(query: str, k: int = 4) -> str:
    """Поиск по базе знаний: методические указания к лабораторным, описание стенда
    и приборов (их сопротивления), правила оформления отчёта по ГОСТ и схем по ЕСКД.
    query — вопрос или ключевые слова по-русски, k — сколько фрагментов вернуть."""
    log.info("search_docs(%r)", query)
    return toolbox().search_docs(query, k)


@server.tool(description="Рассчитать электрическую цепь (постоянный ток, частотные характеристики, "
                         "переходный процесс). Автоматически рисует схему по ГОСТ и графики. "
                         "Возвращает run_id и показания приборов.\n\n" + CIRCUIT_FORMAT)
def simulate_circuit(circuit: dict | str) -> str:
    log.info("simulate_circuit")
    return toolbox().simulate_circuit(circuit)


@server.tool()
def draw_schematic(circuit: dict | str) -> str:
    """Нарисовать принципиальную схему по ГОСТ (без расчёта). Формат circuit — как у simulate_circuit."""
    log.info("draw_schematic")
    return toolbox().draw_schematic(circuit)


def build_report(run_id: str, lab_number: str, lab_title: str, variant: str = "", goal: str = "",
                 theory: str = "", conclusions: str = "") -> str:
    """Собрать отчёт (docx и pdf) по результатам simulate_circuit. Титульный лист, схемы,
    формулы, таблицы и оформление добавляются автоматически.
    lab_number — номер работы, lab_title — тема (например «Цепи постоянного тока»), variant — вариант.
    Для лабораторной №1 (цепи постоянного тока) остальное можно не заполнять: расчёт по Кирхгофу,
    погрешности и вывод формируются сами. Для других работ: goal — цель, theory — краткая теория,
    conclusions — вывод с числами."""
    log.info("build_report(%s)", run_id)
    return toolbox().build_report_text(run_id=run_id, lab_number=lab_number, lab_title=lab_title,
                                       variant=variant, goal=goal, conclusions=conclusions, theory=theory)


def run_python(code: str, run_id: str = "") -> str:
    """Выполнить Python-код в изолированной песочнице (без сети, без доступа к файлам компьютера,
    лимит 60 с). Используй для расчётов по методичке и графиков.
    Внутри: /data/circuit.json и /data/sim.json — результаты simulate_circuit для run_id
    (sim.json: {"analyses": {"op": {"nodes": {узел: потенциал}, "readings": {прибор: показание}},
                             "tran": {"time": [...], "readings": {прибор: [...]}},
                             "ac": {"freq": [...], "readings": {прибор: [[Re, Im], ...]}}}});
    /work — рабочая папка: графики сохраняй туда (plt.savefig("/work/имя.png")), потом их можно
    вставить в отчёт. /data — ТОЛЬКО ЧТЕНИЕ: свои файлы пиши только в /work.
    Доступны numpy, scipy, matplotlib и решатель цепей. Пример:
      import json
      from labagent.circuit import load_circuit
      from labagent.solver import operating_point
      c = load_circuit('/data/circuit.json')          # путь, dict или JSON-строка
      op = operating_point(c)                          # → {"nodes", "currents", "readings"}
      ideal = operating_point(c.idealized())           # те же приборы, но идеальные
      print(json.dumps(op["readings"]))
    Результаты печатай через print — ты увидишь только stdout."""
    log.info("run_python(%s): %d символов", run_id, len(code))
    return toolbox().run_python(code, run_id)


def build_custom_report(run_id: str, lab_number: str, lab_title: str, sections: list | str,
                        variant: str = "", key_results: dict | str | None = None) -> str:
    """Собрать отчёт (docx и pdf) из твоих разделов; титульный лист добавляется сам.
    sections — список разделов по порядку, каждый: {"title": "Описание схемы",
      "text": "текст раздела (абзацы через \\n)", "image": "имя.png" (необязательно: файл из /work
      или schematic.png — схема), "table": {"headers": [...], "rows": [[...], ...]} (необязательно)}.
    Разделы бери из «Содержания отчёта» методички. key_results — главные численные результаты
    в единицах СИ, например {"x0": 21.25, "x_measured": 21.22, "r": 750, "delta": 0.03, "delta0": 0.0318}."""
    log.info("build_custom_report(%s)", run_id)
    return toolbox().build_custom_report_text(run_id=run_id, lab_number=lab_number, lab_title=lab_title,
                                              sections=sections, variant=variant, key_results=key_results)


# регистрируем только инструменты выбранного режима: модель не должна видеть лишнего
if UNIVERSAL:
    server.tool()(run_python)
    server.tool()(build_custom_report)
else:
    server.tool()(build_report)


def main():
    server.run()


if __name__ == "__main__":
    main()
