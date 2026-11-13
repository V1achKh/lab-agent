"""Фейковый OpenAI-совместимый сервер для тестов без видеокарты.

Отвечает по сценарию: проверяет «проводку» проекта (RAG, пайплайн с самоисправлением,
агент с MCP), а не качество модели. Запуск: python tests/fake_llm.py 8765
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

DIM = 256

GOOD = {"title": "Вариант 50", "elements": [  # схема варианта 50 из задания
    {"name": "E2", "type": "V", "n1": "1", "n2": "0", "value": 15},
    {"name": "R1", "type": "R", "n1": "1", "n2": "2", "value": "1 кОм"},
    {"name": "R2", "type": "R", "n1": "1", "n2": "0", "value": "2k"},
    {"name": "E1", "type": "V", "n1": "2", "n2": "0", "value": 10},
    {"name": "R3", "type": "R", "n1": "2", "n2": "3", "value": "3k"},
    {"name": "R4", "type": "R", "n1": "3", "n2": "0", "value": "1k"},
    {"name": "J", "type": "I", "n1": "0", "n2": "3", "value": "25m"},
    {"name": "PV1", "type": "VM", "n1": "2", "n2": "0", "r": "500 кОм"}],
    "analyses": [{"type": "op"}]}
BAD = {"title": "Делитель", "elements": [  # нет номинала у R2 и нет земли
    {"name": "E1", "type": "V", "n1": "1", "n2": "2", "value": 10},
    {"name": "R1", "type": "R", "n1": "1", "n2": "2"}]}


R4_VARIANT = json.loads(json.dumps(GOOD))
R4_VARIANT["title"] = "Вариант 50а"
R4_VARIANT["elements"][-1]["n1"] = "3"
AM_VARIANT = json.loads(json.dumps(GOOD))
AM_VARIANT["title"] = "Вариант 50б"
AM_VARIANT["elements"] = [e for e in GOOD["elements"] if e["name"] not in ("R3", "PV1")] + [
    {"name": "PA1", "type": "AM", "n1": "2", "n2": "4", "r": 10},
    {"name": "R3", "type": "R", "n1": "4", "n2": "3", "value": "3k"}]


RC_VARIANT = {"title": "RC-цепь, вариант 7", "elements": [
    {"name": "E1", "type": "V", "n1": "1", "n2": "0", "value": 5, "wave": "step"},
    {"name": "R1", "type": "R", "n1": "1", "n2": "2", "value": "10k"},
    {"name": "C1", "type": "C", "n1": "2", "n2": "0", "value": "10u"},
    {"name": "PV1", "type": "VM", "n1": "2", "n2": "0", "r": "1M"}],
    "analyses": [{"type": "tran", "t_stop": 0.5}]}

CODE_LAB3 = """
import json
sim = json.load(open('/data/sim.json'))
t = sim['analyses']['tran']['time']; u = sim['analyses']['tran']['readings']['PV1']
target = u[-1] * 0.632
tau = next(t[k] for k in range(len(u)) if u[k] >= target)
R, C, RV = 10e3, 10e-6, 1e6
tau_calc = R * C; tau_v = C * R * RV / (R + RV)
print(json.dumps({"tau": tau, "tau_calc": tau_calc, "tau_with_voltmeter": tau_v}))
"""


def task_text(msgs):
    first = next(m for m in msgs if m["role"] == "user")["content"]
    return first if isinstance(first, str) else " ".join(p.get("text", "") for p in first if isinstance(p, dict))


def _known_images():
    """Картинки заданий из tasks/*.json → эталонная цепь (так фейковая «модель» «видит» картинку)."""
    import base64
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "tasks"
    out = {}
    for f in list(root.glob("*.json")) + list(root.glob("extra/*.json")):
        spec = json.loads(f.read_text(encoding="utf-8"))
        files = spec.get("files", []) + ([spec["image"]] if spec.get("image") else [])
        for x in files:
            p = f.parent / x
            if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.exists():
                out[hashlib.md5(base64.b64encode(p.read_bytes())).hexdigest()] = spec["reference"]
    return out


KNOWN = _known_images()


def task_images(msgs):
    first = next(m for m in msgs if m["role"] == "user")["content"]
    if isinstance(first, str):
        return []
    return [p["image_url"]["url"].split("base64,", 1)[1] for p in first
            if isinstance(p, dict) and p.get("type") == "image_url"]


def pick_circuit(msgs):
    """Фейковая «модель» «смотрит» на картинку (по хэшу узнаёт задание) или читает текст."""
    for b64 in task_images(msgs):
        ref = KNOWN.get(hashlib.md5(b64.encode()).hexdigest())
        if ref is not None:
            return json.loads(json.dumps(ref))
    full = task_text(msgs)
    text = full.split("\nЗадание:\n")[-1].split("Методические указания к этой работе")[0]  # только задание
    if "Переходные процессы" in text or "Лабораторная работа №3" in text:
        assert "Методические указания к этой работе" in full, "методичка не попала в задание"
        return RC_VARIANT
    if "амперметром" in text:
        return AM_VARIANT
    if "резисторе R4" in text:
        return R4_VARIANT
    return GOOD


# код, который «пишет» универсальный агент: первая версия с ошибкой, вторая исправленная
CODE_BUGGY = "import json\nc = json.load(open('/data/circuit.json'))\nprint(undefined_name)\n"
CODE_OK = """
import json, math
from labagent.circuit import load_circuit, Element, Circuit
from labagent.solver import operating_point
c = load_circuit(json.load(open('/data/circuit.json')))
sim = json.load(open('/data/sim.json'))
m = c.meters()[0]
x_meas_full = sim['analyses']['op']['readings'][m.name]
x_meas = round(x_meas_full, 3 - int(math.floor(math.log10(abs(x_meas_full)))))
x0 = operating_point(c.idealized())['readings'][m.name]
# r: прибор убрать, ЭДС -> перемычки (0 В), источники тока -> разрыв, подать 1 А
els = []
for e in c.elements:
    if e.name == m.name or e.type == 'I':
        continue
    e = e.model_copy()
    if e.type == 'V':
        e.value = 0.0
    els.append(e)
els.append(Element(name='Jt', type='I', n1=m.n2, n2=m.n1, value=1.0))
op = operating_point(Circuit.model_construct(title='r', elements=els, analyses=[]))
u = lambda n: 0.0 if n == '0' else op['nodes'][n]
r = max(u(m.n1) - u(m.n2), 0.0)
delta_rel = r / m.r if m.type == 'VM' else (m.r / r if r else 0.0)
delta0 = delta_rel * abs(x_meas)
delta = abs(x_meas - x0)
print(json.dumps({"x0": x0, "x_measured": x_meas, "r": r, "delta": delta, "delta0": delta0}))
"""

def embed(text: str) -> list[float]:
    v = np.zeros(DIM)
    for w in re.findall(r"[\wё]+", text.lower()):
        h = int(hashlib.md5(w[:6].encode()).hexdigest(), 16)
        v[h % DIM] += 1
    n = np.linalg.norm(v) or 1
    return (v / n).tolist()


def tool_call(name, args, k):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{k}", "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}


def universal_step(msgs):
    done = [m for m in msgs if m.get("role") == "tool"]
    n = len(done)
    run_id = None
    for m in done:
        found = re.search(r"run_id: (\S+)", m["content"])
        if found:
            run_id = found.group(1)
    if n == 0:
        return tool_call("search_docs", {"query": "программа выполнения работы содержание отчёта"}, n)
    if n == 1:
        return tool_call("simulate_circuit", {"circuit": pick_circuit(msgs)}, n)
    if n == 2:
        return tool_call("run_python", {"code": CODE_BUGGY, "run_id": run_id}, n)
    lab3 = pick_circuit(msgs) is RC_VARIANT
    if n == 3:
        assert "NameError" in done[-1]["content"], done[-1]["content"]
        return tool_call("run_python", {"code": CODE_LAB3 if lab3 else CODE_OK, "run_id": run_id}, n)
    if n == 4 and lab3:
        res = json.loads(re.search(r"(\{.*\})", done[-1]["content"]).group(1))
        sections = [
            {"title": "Схема исследуемой цепи", "image": "schematic.png"},
            {"title": "График переходного процесса", "image": "plot_tran.png"},
            {"title": "Расчёт постоянной времени",
             "text": f"τ по графику = {res['tau']:.4f} с, τ = RC = {res['tau_calc']:.4f} с, "
                     f"с учётом вольтметра τ' = {res['tau_with_voltmeter']:.4f} с."},
            {"title": "Выводы", "text": "Экспериментальная τ совпадает с τ' с учётом вольтметра."}]
        return tool_call("build_custom_report", {"run_id": run_id, "lab_number": "3",
                                                 "lab_title": "Переходные процессы в RC-цепи", "variant": "7",
                                                 "sections": sections, "key_results": {"tau": res["tau"]}}, n)
    if n == 4:
        out = done[-1]["content"]
        res = json.loads(re.search(r"(\{.*\})", out).group(1))
        sections = [
            {"title": "Полученное задание", "image": "schematic.png"},
            {"title": "Расчёт", "text": f"U0 = {res['x0']:.4f}. Сопротивление схемы r = {res['r']:.1f} Ом."},
            {"title": "Погрешность", "text": f"Δ = {res['delta']:.4g}, Δ0 = {res['delta0']:.4g}.",
             "table": {"headers": ["Величина", "Значение"], "rows": [["U0", res["x0"]], ["Δ0", res["delta0"]]]}},
        ]
        return tool_call("build_custom_report", {"run_id": run_id, "lab_number": "1",
                                                 "lab_title": "Цепи постоянного тока", "variant": "50",
                                                 "sections": sections, "key_results": res}, n)
    return {"role": "assistant", "content": "Готово: " + done[-1]["content"].splitlines()[0]}


def chat(payload: dict) -> dict:
    msgs = payload["messages"]
    system = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
    tools = payload.get("tools")
    if tools:
        names = [t["function"]["name"] for t in tools]
        if "ohm_law" in names:
            return tool_call("ohm_law", {"u": 10, "r": 5}, 0)
        if "run_python" in names:
            return universal_step(msgs)
        done = [m for m in msgs if m.get("role") == "tool"]
        n = len(done)
        if n == 0:
            return tool_call("search_docs", {"query": "лабораторная работа 1 цепи постоянного тока программа"}, n)
        if n == 1:
            return tool_call("search_docs", {"query": "содержание отчёта погрешность измерения"}, n)
        if n == 2:
            return tool_call("simulate_circuit", {"circuit": BAD}, n)
        if n == 3:
            assert "ОШИБКА" in done[-1]["content"], done[-1]["content"]
            # модели иногда присылают цепь строкой — сервер должен это пережить
            return tool_call("simulate_circuit", {"circuit": json.dumps(pick_circuit(msgs), ensure_ascii=False)}, n)
        if n == 4:
            run_id = re.search(r"run_id: (\S+)", done[-1]["content"]).group(1)
            return tool_call("build_report", {
                "run_id": run_id, "lab_number": "1", "lab_title": "Цепи постоянного тока", "variant": "50"}, n)
        return {"role": "assistant", "content": "Отчёт готов: " + done[-1]["content"].splitlines()[0]}
    if "составь описание цепи" in system:
        tries = sum(1 for m in msgs if m["role"] == "assistant")
        circuit = BAD if tries == 0 else pick_circuit(msgs)
        if tries == 1 and circuit is GOOD:   # как Gemma 12B: вольтметр на R3 вместо E1 — код должен поймать
            circuit = json.loads(json.dumps(GOOD))
            circuit["elements"][-1].update(n1="2", n2="3")
        body = {"lab_number": "1", "lab_title": "Цепи постоянного тока", "variant": "50", "circuit": circuit}
        return {"role": "assistant", "content": "<think>думаю…</think>```json\n" + json.dumps(body, ensure_ascii=False) + "\n```"}
    if "текстовые части отчёта" in system:
        return {"role": "assistant", "content": json.dumps({
            "goal": "Изучить работу делителя напряжения.",
            "theory": "Ток I = E/(R1+R2), напряжение U2 = E·R2/(R1+R2).",
            "conclusions": "Вольтметр с входным сопротивлением 1 МОм занижает напряжение на 4,76 %."}, ensure_ascii=False)}
    if "проверяешь отчёт студента" in system:
        assert "Требования:" in msgs[-1]["content"] and "Отчёт:" in msgs[-1]["content"]
        return {"role": "assistant", "content": json.dumps({"remarks": [
            {"rule": "Вывод опирается на полученные числа", "where": "Вывод",
             "comment": "(фейковая модель) добавьте сравнение с расчётом"}]}, ensure_ascii=False)}
    return {"role": "assistant", "content": "Москва"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.endswith("/models"):
            return self._send({"data": [{"id": "fake"}, {"id": "fake-embed"}]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            inp = payload["input"]
            inp = [inp] if isinstance(inp, str) else inp
            return self._send({"data": [{"index": i, "embedding": embed(t)} for i, t in enumerate(inp)]})
        if self.path.endswith("/chat/completions"):
            msg = chat(payload)
            return self._send({"choices": [{"message": msg, "finish_reason": "stop"}],
                               "usage": {"total_tokens": 100}})
        self._send({"error": "not found"}, 404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
