"""Командная строка: python -m labagent <команда>.

  check                     проверить установку (ngspice, LibreOffice, модель, эмбеддинги)
  index                     проиндексировать документы из data/docs
  search "запрос"           посмотреть, что находит RAG
  ask "вопрос" [--no-rag]   ответ модели по документам (или без них — для сравнения)
  draw cirсuit.json         нарисовать схему (без модели)
  simulate circuit.json     посчитать цепь, нарисовать схему и графики (без модели)
  demo                      полные отчёты по эталонным цепям из tasks/ (без модели)
  pipeline ФАЙЛЫ|ПАПКА [--variant N]  жёсткий пайплайн: модель только составляет цепь (и текст)
  agent ФАЙЛЫ|ПАПКА [--variant N]     агент с MCP-инструментами
  agent ... --universal     универсальный агент: сам пишет код расчёта (выполняется в песочнице)
      ФАЙЛЫ — картинка варианта, методичка PDF, таблица вариантов…: что есть что, программа
      определяет по содержимому; --variant выбирает строку таблицы / страницу / картинку
  review report.docx        проверить отчёт по списку замечаний преподавателя
  sandbox build|test        собрать образ песочницы / проверить изоляцию
  compare                   сравнить пайплайн и агента на заданиях из tasks/
  mcp                       запустить MCP-сервер (для подключения к другим клиентам)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import config


def _prepare(args, llm):
    """Что подано на вход → (текст для модели, картинки, индекс методички).

    Аргументы — файлы, папки и (необязательно) текст задания, в любом порядке:
      python -m labagent pipeline вариант.png Методичка_Лаб1.pdf
      python -m labagent pipeline ~/лабы/лаб2/ --variant 32
    Файлы разбираются по содержимому: методичка — где есть «Программа выполнения работы» /
    «Содержание отчёта», остальное — задание (картинки схем, таблицы вариантов).
    tasks/*.json (ключи ответов для compare) тоже понимаются."""
    from .inputs import InputError, classify, prepare, spec_files
    from .manual import ManualError

    texts, paths, files, manuals, variant = [], [], [], [], args.variant
    for item in args.items:
        pth = Path(item).expanduser()
        if pth.suffix == ".json" and pth.is_file():
            spec = json.loads(pth.read_text(encoding="utf-8"))
            if spec.get("task"):
                texts.append(spec["task"])
            f, m = spec_files(spec, pth.parent)
            files += f
            manuals += m
            variant = variant or spec.get("variant")
        elif pth.exists():
            paths.append(pth)
        else:
            texts.append(item)
    try:
        t, m = classify(paths + [Path(x) for x in (args.input or []) + (args.image or [])])
        files += [str(x) for x in t]
        manuals += [str(x) for x in m] + list(args.manual or [])
        if not files and not texts:
            sys.exit("Нечего делать: укажите файлы задания (картинку варианта, PDF) и/или текст задания.")
        return prepare(" ".join(texts), files, manuals, args.text_only, llm, variant=variant)
    except (InputError, ManualError) as exc:
        sys.exit(f"Ошибка во входных данных: {exc}")


def cmd_check(args):
    import shutil

    from . import spice
    from .llm import LLM, LLMError
    from .simulate import cutoff_frequency, time_constant

    ok = True
    print(f"Модель:      {config.LLM_MODEL} @ {config.LLM_URL}")
    print(f"Эмбеддинги:  {config.EMBED_MODEL} @ {config.EMBED_URL}")

    # ngspice + сверка со встроенным решателем
    if spice.available():
        from . import solver

        from .selftest import circuits

        c, c2, c3 = circuits()
        try:
            a = spice.run(c, c.analyses[0])["readings"]
            b = solver.run(c, c.analyses[0])["readings"]
            d_op = max(abs(a[k] - b[k]) / abs(b[k]) for k in b)
            a2 = spice.run(c2, c2.analyses[0])
            b2 = solver.run(c2, c2.analyses[0])
            d_fc = abs(cutoff_frequency(a2["freq"], a2["readings"]["PV1"]) /
                       cutoff_frequency(b2["freq"], b2["readings"]["PV1"]) - 1)
            a3 = spice.run(c3, c3.analyses[0])
            b3 = solver.run(c3, c3.analyses[0])
            d_tau = abs(time_constant(a3["time"], a3["readings"]["PV1"]) /
                        time_constant(b3["time"], b3["readings"]["PV1"]) - 1)
            good = d_op < 1e-4 and d_fc < 0.01 and d_tau < 0.01
            ok &= good
            print(f"[{'OK' if good else '!!'}] ngspice: расхождение со встроенным решателем — "
                  f"постоянный ток {d_op:.1e}, частота среза {d_fc:.1e}, τ {d_tau:.1e}")
        except Exception as exc:
            ok = False
            print(f"[!!] ngspice установлен, но не отработал: {exc}")
    else:
        print("[--] ngspice не найден — будет использован встроенный решатель (sudo pacman -S ngspice)")

    from . import sandbox

    sb = sandbox.backend()
    print(f"[{'OK' if sb else '--'}] песочница для универсального режима: "
          f"{sandbox.describe() if sb else 'не найдена (sudo pacman -S podman; python -m labagent sandbox build)'}")

    soffice = shutil.which(config.SOFFICE_BIN) or shutil.which("libreoffice")
    print(f"[{'OK' if soffice else '--'}] LibreOffice: {soffice or 'не найден — отчёт будет только в .docx'}")

    try:
        import urllib.request

        with urllib.request.urlopen(config.LLM_URL.rstrip("/") + "/models", timeout=10) as r:
            names = [m.get("id") for m in json.loads(r.read()).get("data", [])]
        print(f"[OK] сервер модели отвечает, доступные модели: {', '.join(names[:12]) or '—'}")
        if names and config.LLM_MODEL not in names:
            print(f"[!!] модели «{config.LLM_MODEL}» нет в списке — проверьте LAB_LLM_MODEL")
    except Exception as exc:
        ok = False
        print(f"[!!] сервер модели не отвечает: {exc}")

    llm = LLM()
    try:
        t0 = time.time()
        reply = llm.ask("Отвечай одним словом.", "Столица России?", max_tokens=400)
        print(f"[OK] чат: «{reply[:60]}» за {time.time() - t0:.1f} с")
    except LLMError as exc:
        ok = False
        print(f"[!!] чат: {exc}")
    if args.tools:
        tool = [{"type": "function", "function": {
            "name": "ohm_law", "description": "Вычислить ток по закону Ома",
            "parameters": {"type": "object", "properties": {"u": {"type": "number"}, "r": {"type": "number"}},
                           "required": ["u", "r"]}}}]
        try:
            msg = llm.chat([{"role": "user", "content": "Какой ток при U=10 В и R=5 Ом? Используй инструмент."}],
                           tools=tool)
            calls = msg.get("tool_calls") or []
            good = bool(calls) and calls[0]["function"]["name"] == "ohm_law"
            ok &= good
            print(f"[{'OK' if good else '!!'}] вызов инструментов: "
                  f"{calls[0]['function'] if calls else 'модель не вызвала инструмент: ' + msg.get('content', '')[:80]}")
        except LLMError as exc:
            ok = False
            print(f"[!!] вызов инструментов: {exc}")
    try:
        v = llm.embed(["проверка эмбеддингов"])
        print(f"[OK] эмбеддинги: размерность {v.shape[1]}")
    except LLMError as exc:
        ok = False
        print(f"[!!] эмбеддинги: {exc}")
    print("\nВсё готово." if ok else "\nЕсть проблемы — см. строки с [!!].")


def cmd_index(args):
    from .llm import LLM
    from .rag import RagIndex

    t0 = time.time()
    idx = RagIndex.build(LLM())
    srcs = sorted({c.source for c in idx.chunks})
    print(f"Проиндексировано: {len(idx.chunks)} фрагментов из {len(srcs)} файлов за {time.time() - t0:.1f} с")
    for s in srcs:
        print("  •", s)


def cmd_search(args):
    from .llm import LLM
    from .rag import RagIndex

    idx = RagIndex.load()
    llm = None if args.mode == "bm25" else LLM()
    for i, r in enumerate(idx.search(args.query, llm, k=args.k, mode=args.mode), 1):
        cos = f", косинус {r['cosine']}" if r["cosine"] is not None else ""
        print(f"\n[{i}] {r['source']} — {r['heading']} (RRF {r['score']}{cos})\n{r['text'][:400]}")


def cmd_ask(args):
    from .llm import LLM
    from .prompts import NO_RAG_SYSTEM, RAG_QA_SYSTEM
    from .rag import RagIndex, format_context

    llm = LLM()
    if args.no_rag:
        print(llm.ask(NO_RAG_SYSTEM, args.question))
        return
    from .manual import attach_manuals, load_rag

    _, task_index = attach_manuals("", args.manual, llm)
    rag = load_rag(task_index) if task_index else RagIndex.load()
    res = rag.search(args.question, llm, k=args.k)
    print(llm.ask(RAG_QA_SYSTEM, f"Фрагменты:\n{format_context(res)}\n\nВопрос: {args.question}"))
    print("\nИсточники:", ", ".join(f"[{i}] {r['source']}" for i, r in enumerate(res, 1)))


def _load_circuit_file(path: str):
    from .circuit import load_circuit

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_circuit(data.get("reference", data))


def cmd_draw(args):
    from .schematic import draw_schematic

    c = _load_circuit_file(args.file)
    out = Path(args.out or Path(args.file).with_suffix(".png"))
    draw_schematic(c, out, out.with_suffix(".svg"))
    print(f"Схема: {out}")


def cmd_simulate(args):
    from .tools import Toolbox

    r = Toolbox().simulate(_load_circuit_file(args.file))
    if not r["ok"]:
        sys.exit("Ошибка: " + r["error"])
    print(r["summary"])
    print(f"\nФайлы: {r['run_dir']}")


def cmd_demo(args):
    """Полный отчёт без модели: эталонная цепь + тексты-заглушки. Проверка, что всё, кроме LLM, работает."""
    from .tools import Toolbox

    tb = Toolbox()
    for tf in sorted((config.ROOT / "tasks").glob("*.json")):
        spec = json.loads(tf.read_text(encoding="utf-8"))
        r = tb.simulate(spec["reference"])
        rep = tb.build_report(
            run_id=r["run_id"], lab_number=str(spec.get("lab_number", "—")),
            lab_title=spec.get("lab_title") or r["circuit"].title, variant=str(spec.get("variant", "")),
            goal="(цель работы напишет модель)", theory="(теорию напишет модель)",
            conclusions="" if r["profile"] == "dc_lab" else "(выводы напишет модель по результатам:)\n" + r["summary"])
        print(f"{tf.stem}: {rep['pdf'] or rep['docx']}")


def _review(res, llm, args):
    if getattr(args, "no_review", False) or not res.get("report_docx"):
        return
    from .review import print_review, review_report

    print_review(review_report(res["report_docx"], llm=llm))


def cmd_pipeline(args):
    from .llm import LLM
    from .manual import load_rag
    from .pipeline import run_pipeline

    llm = LLM()
    task, images, task_index = _prepare(args, llm)
    rag = load_rag(task_index)
    if rag is None:
        print("Внимание: база знаний пуста (python -m labagent index). Работаю без RAG.")
    res = run_pipeline(task, llm=llm, rag=rag, images=images)
    print(f"\nГотово за {res.get('seconds')} с, вызовов модели: {res.get('llm_calls')}" if res["ok"]
          else f"\nНе получилось: {res.get('error')}")
    _review(res, llm, args)


def cmd_agent(args):
    from .agent import run_agent
    from .llm import LLM

    llm = LLM()
    task, images, task_index = _prepare(args, llm)
    if task_index and not args.universal:
        print("Подсказка: готовая процедура есть только для лабораторной №1. Для других работ "
              "запускайте с --universal — тогда агент сам выполнит расчёты по методичке.")
    steps = args.max_steps or (24 if args.universal else 14)
    res = run_agent(task, llm=llm, max_steps=steps, images=images, universal=args.universal,
                    task_index=task_index)
    if res.get("error"):
        print(f"\nАгент остановлен: {res['error']}")
    if res.get("fallback"):
        print("Отчёт собран кодом по последней цепи модели (агент сам до отчёта не дошёл) — "
              "сверьте схему в отчёте с заданием.")
    if res.get("error") and not res.get("report_docx"):
        sys.exit(1)
    print("\nОтвет агента:\n" + (res["answer"] or "—"))
    print(f"\nОтчёт: {res['report_pdf'] or res['report_docx'] or 'не создан'}; "
          f"шагов: {res['steps']}, вызовов модели: {res['llm_calls']}, время: {res['seconds']} с")
    _review(res, llm, args)


def cmd_review(args):
    from .llm import LLM
    from .review import print_review, review_report

    print_review(review_report(args.report, llm=None if args.no_llm else LLM()))


def cmd_compare(args):
    from .compare import run_compare

    tasks = [Path(t) for t in args.tasks] if args.tasks else sorted((config.ROOT / "tasks").glob("*.json"))
    md = run_compare(args.modes.split(","), args.repeats, tasks, text_only=args.text_only)
    print(f"\nТаблица сохранена: {md}")


def cmd_sandbox(args):
    import tempfile

    from . import sandbox

    if args.action == "build":
        print(sandbox.build_image())
        return
    print(f"Песочница: {sandbox.describe()}")
    with tempfile.TemporaryDirectory(prefix="labsbx_test_") as tmp:
        results = sandbox.self_test(Path(tmp))
    for title, ok, detail in results:
        print(f"[{'OK' if ok else '!!'}] {title}: {detail}")
    if all(ok for _, ok, _ in results):
        print("\nИзоляция работает.")
    else:
        print("\nЕсть проблемы — см. [!!].")
        sys.exit(1)


def cmd_mcp(args):
    from .mcp_server import main

    main()


def _input_args(s):
    s.add_argument("items", nargs="*", metavar="ФАЙЛ|ПАПКА|ТЕКСТ",
                   help="файлы и папки лабы (картинки вариантов, методичка PDF, таблица вариантов) и/или текст "
                        "задания; разбираются по содержимому")
    s.add_argument("--variant", "-v", help="номер варианта: из таблицы вариантов берётся только его строка, "
                                           "из нескольких картинок — та, где номер в имени файла")
    s.add_argument("--input", "-i", action="append", metavar="ФАЙЛ",
                   help="файл задания, если нужно указать явно (можно несколько раз)")
    s.add_argument("--image", action="append", metavar="ФАЙЛ", help="то же, что --input (для совместимости)")
    s.add_argument("--manual", "-m", action="append", metavar="ФАЙЛ",
                   help="методичка к работе: PDF, DOCX, MD, TXT (можно несколько)")
    s.add_argument("--text-only", action="store_true", help="не отправлять модели картинки")
    s.add_argument("--no-review", action="store_true", help="не проверять готовый отчёт по списку замечаний")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m labagent", description="Агент для лабораторных по электротехнике")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check", help="проверить установку")
    s.add_argument("--tools", action="store_true", help="проверить и вызов инструментов моделью")
    s.set_defaults(fn=cmd_check)
    sub.add_parser("index", help="проиндексировать data/docs").set_defaults(fn=cmd_index)
    s = sub.add_parser("search", help="поиск по базе знаний")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=4)
    s.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    s.set_defaults(fn=cmd_search)
    s = sub.add_parser("ask", help="вопрос по документам")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=4)
    s.add_argument("--no-rag", action="store_true", help="спросить модель без документов (для сравнения)")
    s.add_argument("--manual", action="append", help="искать ещё и в этой методичке (можно несколько)")
    s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("draw", help="нарисовать схему из JSON")
    s.add_argument("file")
    s.add_argument("-o", "--out")
    s.set_defaults(fn=cmd_draw)
    s = sub.add_parser("simulate", help="посчитать цепь из JSON")
    s.add_argument("file")
    s.set_defaults(fn=cmd_simulate)
    sub.add_parser("demo", help="отчёты по эталонным цепям без модели").set_defaults(fn=cmd_demo)
    s = sub.add_parser("pipeline", help="жёсткий пайплайн")
    _input_args(s)
    s.set_defaults(fn=cmd_pipeline)
    s = sub.add_parser("agent", help="агент с MCP")
    _input_args(s)
    s.add_argument("--universal", action="store_true",
                   help="универсальный режим: без готовых процедур, агент сам пишет код расчёта (в песочнице)")
    s.add_argument("--max-steps", type=int, default=0)
    s.set_defaults(fn=cmd_agent)
    s = sub.add_parser("compare", help="сравнить режимы")
    s.add_argument("--modes", default="pipeline,agent,universal")
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--text-only", action="store_true", help="не передавать модели картинки схем")
    s.add_argument("tasks", nargs="*")
    s.set_defaults(fn=cmd_compare)
    s = sub.add_parser("sandbox", help="песочница для кода модели: build — собрать образ, test — проверить изоляцию")
    s.add_argument("action", choices=["build", "test"])
    s.set_defaults(fn=cmd_sandbox)
    s = sub.add_parser("review", help="проверить готовый отчёт по списку замечаний преподавателя")
    s.add_argument("report", help="report.docx (свой или созданный агентом)")
    s.add_argument("--no-llm", action="store_true", help="только автоматические проверки, без модели")
    s.set_defaults(fn=cmd_review)
    sub.add_parser("mcp", help="запустить MCP-сервер (stdio)").set_defaults(fn=cmd_mcp)
    args = p.parse_args(argv)
    from .llm import LLMError

    try:
        args.fn(args)
    except LLMError as exc:   # сервер модели не запущен и т.п. — понятное сообщение вместо трассировки
        msg = str(exc)
        if "Connection refused" in msg or "не удалось подключиться" in msg:
            port = "8081 (эмбеддинги)" if ":8081" in msg else "8080 (чат-модель)"
            sys.exit(f"\nМодель не запущена: никто не слушает порт {port}.\n"
                     "Запустите в отдельном терминале: cd ~/lab-agent && scripts/llamacpp.sh gemma12\n"
                     "и дождитесь строки «listening on http://127.0.0.1:8080».")
        sys.exit(f"\nОшибка модели: {msg}")


if __name__ == "__main__":
    main()
