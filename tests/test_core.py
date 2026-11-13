import math
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labagent.circuit import Circuit, parse_value  # noqa: E402

needs_files = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "tasks/lab1/Методичка_Лаб1.pdf").exists(),
    reason="нет файлов преподавателя в tasks/")
from labagent.llm import extract_json  # noqa: E402
from labagent.schematic import draw_schematic  # noqa: E402
from labagent.simulate import cutoff_frequency, simulate, time_constant  # noqa: E402


def c(d):
    return Circuit.model_validate(d)


def test_parse_value():
    assert parse_value("4.7k") == 4700
    assert parse_value("2,2 мкФ") == 2.2e-6
    assert parse_value("1 МОм") == 1e6
    assert parse_value("10m") == 0.01
    assert parse_value("100 нФ") == 1e-7


def test_divider_with_real_voltmeter():
    sim = simulate(c({"elements": [
        {"name": "E1", "type": "V", "n1": "1", "n2": "0", "value": 10},
        {"name": "R1", "type": "R", "n1": "1", "n2": "2", "value": "100k"},
        {"name": "R2", "type": "R", "n1": "2", "n2": "0", "value": "100k"},
        {"name": "PV1", "type": "VM", "n1": "2", "n2": "0", "r": "1M"}]}), engine="builtin")
    rp = 100e3 * 1e6 / 1.1e6
    assert abs(sim["analyses"]["op"]["readings"]["PV1"] - 10 * rp / (100e3 + rp)) < 1e-5
    assert abs(sim["ideal_op"]["readings"]["PV1"] - 5) < 1e-5


def test_rc_filter_cutoff():
    circ = c({"elements": [
        {"name": "E1", "type": "V", "n1": "1", "n2": "0", "ac": 1},
        {"name": "R1", "type": "R", "n1": "1", "n2": "2", "value": "1k"},
        {"name": "C1", "type": "C", "n1": "2", "n2": "0", "value": "0.1u"},
        {"name": "PV1", "type": "VM", "n1": "2", "n2": "0"}],
        "analyses": [{"type": "ac", "f_start": 10, "f_stop": 1e5, "points_per_decade": 50}]})
    a = simulate(circ, engine="builtin")["analyses"]["ac"]
    fc = cutoff_frequency(a["freq"], a["readings"]["PV1"])
    assert abs(fc - 1 / (2 * math.pi * 1e3 * 1e-7)) / fc < 0.005


def test_rc_transient_tau():
    circ = c({"elements": [
        {"name": "E1", "type": "V", "n1": "1", "n2": "0", "value": 5, "wave": "step"},
        {"name": "R1", "type": "R", "n1": "1", "n2": "2", "value": "10k"},
        {"name": "C1", "type": "C", "n1": "2", "n2": "0", "value": "10u"},
        {"name": "PV1", "type": "VM", "n1": "2", "n2": "0"}],
        "analyses": [{"type": "tran", "t_stop": 0.5}]})
    t = simulate(circ, engine="builtin")["analyses"]["tran"]
    assert abs(time_constant(t["time"], t["readings"]["PV1"]) - 0.1) < 0.002


def test_validation_messages():
    try:
        c({"elements": [{"name": "R1", "type": "R", "n1": "1", "n2": "2", "value": 1}]})
    except Exception as exc:
        msg = str(exc)
        assert "источник" in msg and "земли" in msg
    else:
        raise AssertionError("ожидалась ошибка")


def test_extract_json():
    assert extract_json('<think>хм</think>Вот: ```json\n{"a": {"b": "}"}}\n```')["a"]["b"] == "}"


def test_schematic_renders():
    with tempfile.TemporaryDirectory() as d:
        p = draw_schematic(c({"elements": [
            {"name": "E1", "type": "V", "n1": "1", "n2": "0", "value": 1},
            {"name": "L1", "type": "L", "n1": "1", "n2": "2", "value": "1m"},
            {"name": "R1", "type": "R", "n1": "2", "n2": "0", "value": 1},
            {"name": "PV1", "type": "VM", "n1": "1", "n2": "2"}]}), Path(d) / "s.png")
        assert p.exists() and p.stat().st_size > 5000


V50 = [{"name": "E2", "type": "V", "n1": "1", "n2": "0", "value": 15},
       {"name": "R1", "type": "R", "n1": "1", "n2": "2", "value": "1k"},
       {"name": "R2", "type": "R", "n1": "1", "n2": "0", "value": "2k"},
       {"name": "E1", "type": "V", "n1": "2", "n2": "0", "value": 10},
       {"name": "R3", "type": "R", "n1": "2", "n2": "3", "value": "3k"},
       {"name": "R4", "type": "R", "n1": "3", "n2": "0", "value": "1k"},
       {"name": "J", "type": "I", "n1": "0", "n2": "3", "value": "25m"}]


def test_lab1_variant50():
    from labagent.dc_lab import analyze

    r = analyze(c({"elements": V50 + [{"name": "PV1", "type": "VM", "n1": "2", "n2": "0", "r": "500k"}]}))
    assert abs(r.x0 - 10) < 1e-9 and r.r == 0 and r.delta0 == 0
    assert len(r.k.kcl) == 3 and len(r.k.kvl) == 3
    assert r.refine and r.refine["ok"]


def test_lab1_voltmeter_error():
    from labagent.dc_lab import analyze

    r = analyze(c({"elements": V50 + [{"name": "PV1", "type": "VM", "n1": "3", "n2": "0", "r": "500k"}]}))
    assert abs(r.x0 - 21.25) < 1e-9
    assert abs(r.r - 750) < 1e-6
    assert abs(r.x_disp - 21.22) < 1e-9
    assert r.ok_first and r.delta <= r.delta0


def test_lab1_ammeter():
    from labagent.dc_lab import analyze

    els = [e for e in V50 if e["name"] != "R3"] + [
        {"name": "PA1", "type": "AM", "n1": "2", "n2": "4", "r": 10},
        {"name": "R3", "type": "R", "n1": "4", "n2": "3", "value": "3k"}]
    r = analyze(c({"elements": els}))
    assert abs(r.x0 + 3.75e-3) < 1e-12 and abs(r.r - 4000) < 1e-6
    assert r.ok_first


def test_kirchhoff_matches_solver():
    from labagent.dc_lab import analyze

    r = analyze(c({"elements": V50 + [{"name": "PV1", "type": "VM", "n1": "3", "n2": "0", "r": "500k"}]}))
    I = r.k.solution
    assert abs(-I["R3"] + I["R4"] - 0.025) < 1e-12
    assert abs(3000 * I["R3"] + 1000 * I["R4"] - 10) < 1e-9


def test_universal_numbers_scoring():
    import json as _json

    from labagent.compare import score_numbers

    ref = c({"elements": V50 + [{"name": "PV1", "type": "VM", "n1": "3", "n2": "0", "r": "500k"}]})
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        good = {"x0": 21.25, "x_measured": 21.22, "r": 750, "delta": 0.03, "delta0": 0.03183}
        (run / "key_results.json").write_text(_json.dumps(good))
        assert score_numbers(ref, run)[0]
        (run / "key_results.json").write_text(_json.dumps({**good, "r": 1000}))
        ok, note = score_numbers(ref, run)
        assert not ok and "r =" in note


def test_custom_report_rejects_paths_outside():
    from labagent.tools import Toolbox

    with tempfile.TemporaryDirectory() as d:
        tb = Toolbox(runs_dir=Path(d), universal=True)
        r = tb.simulate({"elements": V50 + [{"name": "PV1", "type": "VM", "n1": "3", "n2": "0", "r": "500k"}]})
        rep = tb.build_custom_report(r["run_id"], "1", "Тест", [
            {"title": "Схема", "image": "schematic.png"},
            {"title": "Чужой файл", "image": "/etc/passwd"}])
        assert rep["ok"] and any("passwd" in p for p in rep["problems"])
        assert "ideal_op" not in r["sim"]


@needs_files
def test_manual_pdf_and_scan_detection():
    from labagent.manual import ManualError, manual_block, read_manual

    root = Path(__file__).resolve().parent.parent
    text = read_manual(root / "tasks" / "lab3" / "Методичка_Лаб3.pdf")
    assert "Содержание отчёта" in text and "Программа выполнения работы" in text
    assert "Методические указания" in manual_block("x.pdf", text)
    with tempfile.TemporaryDirectory() as d:
        empty = Path(d) / "scan.txt"
        empty.write_text("   ")
        try:
            read_manual(empty)
        except ManualError as exc:
            assert "скан" in str(exc)
        else:
            raise AssertionError("ожидалась ошибка")


@needs_files
def test_task_inputs_many_files():
    from labagent.inputs import InputError, load_inputs
    from labagent.llm import user_message

    root = Path(__file__).resolve().parent.parent
    docx = next((root / "examples").glob("*.docx"))
    with tempfile.TemporaryDirectory() as d:
        txt = Path(d) / "данные.txt"
        txt.write_text("E = 5 В, R = 10 кОм")
        ti = load_inputs("Лаб. 3", [root / "tasks" / "lab1" / "вариант_50.png",
                                    root / "tasks" / "lab3" / "Методичка_Лаб3.pdf", docx, txt])
        assert "E = 5 В" in ti.text and "Содержание отчёта" in ti.text
        assert ti.images[0].endswith("вариант_50.png") and len(ti.images) >= 3
        msg = user_message(ti.prompt(), ti.images)
        assert sum(p["type"] == "image_url" for p in msg["content"]) == len(ti.images)
        try:
            load_inputs("x", [Path(d) / "a.xyz"])
        except InputError:
            pass
        else:
            raise AssertionError("ожидалась ошибка")


def test_rag_merge_base_and_manual():
    import numpy as np

    from labagent.rag import Chunk, RagIndex

    base = RagIndex([Chunk("Поля страницы: левое 30 мм", "gost.md", "ГОСТ")], np.eye(2)[:1], "m")
    task = RagIndex([Chunk("Постоянная времени τ = RC", "lab3.pdf", "Теория")], np.eye(2)[1:], "m")
    both = RagIndex.merge(base, None, task)
    assert len(both.chunks) == 2 and both.emb.shape == (2, 2)
    assert both.search("постоянная времени", None, k=1, mode="bm25")[0]["source"] == "lab3.pdf"


def test_review_auto_checks():
    from docx import Document

    from labagent.review import auto_checks, read_report

    with tempfile.TemporaryDirectory() as d:
        doc = Document()
        doc.add_paragraph("Отчёт по лабораторной работе №1")
        doc.add_paragraph("Студент Фамилия И. О.")
        doc.add_paragraph().add_run("Описание схемы:").bold = True
        doc.add_paragraph("Узел 1: левый вывод резистора R1 и вольтметр PV1.")
        doc.add_paragraph().add_run("Измерение:").bold = True
        doc.add_paragraph("Показание 21.22 В и 21,22 В, абсолютная погрешность nan, U = 2*R, "
                          "по ГОСТ 7.32 и NI Multisim 14.0, законы Кирхгофа")
        doc.add_paragraph().add_run("Расчёт:").bold = True
        p = Path(d) / "r.docx"
        doc.save(p)
        rem = " | ".join(auto_checks(read_report(p), p))
        assert "Десятичный разделитель разный" in rem and "«7.32»" not in rem and "«14.0»" not in rem
        assert "nan" in rem and "Пустой раздел «Расчёт:»" in rem and "Нет раздела «Вывод»" in rem
        assert "Фамилия" in rem and "правила Кирхгофа" in rem and "абсолютная погрешность" in rem
        assert "«*»" in rem and "Имя файла «r.docx»" in rem
        assert "Описание схемы: «Узел»" in rem and "Требуется определить" in rem
        assert "заканчиваться фразой" in rem


def test_generated_report_passes_auto_checks():
    import json as _json

    from labagent.review import auto_checks, read_report
    from labagent.tools import Toolbox

    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as d:
        tb = Toolbox(runs_dir=Path(d))
        for name in ("lab1_v50", "lab1_varA", "lab1_varB"):
            spec = _json.loads((root / "tasks" / f"{name}.json").read_text(encoding="utf-8"))
            r = tb.simulate(spec["reference"])
            rep = tb.build_report(run_id=r["run_id"], lab_number="1", lab_title="Цепи постоянного тока",
                                  variant=spec["variant"])
            bad = [x for x in auto_checks(read_report(rep["docx"]), rep["docx"])
                   if "титульного листа" not in x and "Имя файла" not in x]
            assert not bad, (name, bad)
            assert Path(rep["docx"]).name.endswith("Лаб.1.docx")


@needs_files
def test_files_classified_and_variant_picked():
    from labagent.inputs import classify, load_inputs

    root = Path(__file__).resolve().parent.parent / "tasks"
    t, m = classify([root / "lab1", root / "lab2"])
    assert {x.name for x in m} == {"Методичка_Лаб1.pdf", "Методичка_Лаб2.pdf"}
    assert "Варианты_Лаб2.pdf" in {x.name for x in t}
    ti = load_inputs("", [root / "lab2" / "Варианты_Лаб2.pdf"], "32")
    assert "32 LM2931CS 25" in ti.text and "LM109H" not in ti.text and not ti.images
    ti = load_inputs("", sorted((root / "lab1").glob("*.png")), "Б")
    assert [Path(x).name for x in ti.images] == ["вариант_Б.png"]


def test_description_rules():
    from labagent.dc_lab import describe_circuit

    lines = describe_circuit(c({"elements": V50 + [{"name": "PV1", "type": "VM", "n1": "2", "n2": "0", "r": "500k"}]}))
    text = " ".join(lines)
    assert "узел" not in text.lower() and "PV1" not in text and "вольтметр" not in text
    assert lines[-1] == "Требуется определить падение напряжения на источнике напряжения E1."


def test_meter_where_task_says():
    from labagent.dc_lab import analyze, target_mismatch, task_target

    task = "Лабораторная №1. Необходимо определить падение напряжения на источнике напряжения E1."
    good = c({"elements": V50 + [{"name": "PV1", "type": "VM", "n1": "2", "n2": "0", "r": "500k"}]})
    bad = c({"elements": V50 + [{"name": "PV1", "type": "VM", "n1": "2", "n2": "3", "r": "500k"}]})
    assert task_target(task) == ("V", "E1")
    assert target_mismatch(task, good) is None
    msg = target_mismatch(task, bad)
    assert msg and 'n1 = "2", n2 = "0"' in msg
    assert task_target(task + "\n\nМетодические указания к этой работе (файл «x»):\nнапряжение на R9") == ("V", "E1")

    am_task = "Необходимо определить ток через резистор R3 амперметром 10 Ом."
    els = [e for e in V50 if e["name"] != "R3"]
    am_ok = c({"elements": els + [{"name": "PA1", "type": "AM", "n1": "2", "n2": "4", "r": 10},
                                  {"name": "R3", "type": "R", "n1": "4", "n2": "3", "value": "3k"}]})
    assert target_mismatch(am_task, am_ok) is None
    els4 = [e for e in V50 if e["name"] != "R4"]
    am_bad = c({"elements": els4 + [{"name": "PA1", "type": "AM", "n1": "3", "n2": "5", "r": 10},
                                    {"name": "R4", "type": "R", "n1": "5", "n2": "0", "value": "1k"}]})
    assert "не последовательно" in target_mismatch(am_task, am_bad)


def test_structure_problems_and_shorted_element():
    import json as _json

    from labagent.dc_lab import report_blocks, structure_problems, analyze

    root = Path(__file__).resolve().parent.parent
    ref = _json.loads((root / "tasks" / "lab1_varA.json").read_text(encoding="utf-8"))["reference"]
    assert structure_problems(c(ref)) is None
    bad = _json.loads(_json.dumps(ref))
    bad["elements"][3]["r"] = 0
    bad["elements"].append({"name": "R9", "type": "R", "n1": "2", "n2": "4", "value": 100})
    msg = structure_problems(c(bad))
    assert "внутреннее сопротивление" in msg and "закорачивает" in msg
    bad["elements"][3]["r"] = 0.05
    with tempfile.TemporaryDirectory() as d:
        from labagent.schematic import draw_schematic

        draw_schematic(c(bad), Path(d) / "schematic.png")
        assert report_blocks(analyze(c(bad)), d, "builtin")


def test_refine_uses_refined_delta0():
    from labagent.dc_lab import analyze

    r = analyze(c({"elements": V50 + [{"name": "PV1", "type": "VM", "n1": "2", "n2": "3", "r": "500k"}]}))
    assert not r.ok_first and r.refine and r.refine["ok"]
    assert abs(r.refine["delta0"] - r.rel * abs(r.refine["x"])) < 1e-15


def test_decimal_separator_setting():
    from labagent import config
    from labagent.circuit import fmt_si

    old = config.DECIMAL
    try:
        config.DECIMAL = "."
        assert fmt_si(4700, "Ом") == "4.7 кОм"
        config.DECIMAL = ","
        assert fmt_si(4700, "Ом") == "4,7 кОм"
    finally:
        config.DECIMAL = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
