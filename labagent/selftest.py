"""Эталонные цепи для самопроверки (ngspice против встроенного решателя)."""
from .circuit import load_circuit

DC = {"title": "Делитель", "elements": [
    {"name": "E1", "type": "V", "n1": "1", "n2": "0", "value": 10},
    {"name": "PA1", "type": "AM", "n1": "1", "n2": "2", "r": 10},
    {"name": "R1", "type": "R", "n1": "2", "n2": "3", "value": "100k"},
    {"name": "R2", "type": "R", "n1": "3", "n2": "0", "value": "100k"},
    {"name": "J1", "type": "I", "n1": "0", "n2": "3", "value": "10u"},
    {"name": "PV1", "type": "VM", "n1": "3", "n2": "0", "r": "1M"}],
    "analyses": [{"type": "op"}]}
AC = {"title": "RC-фильтр", "elements": [
    {"name": "E1", "type": "V", "n1": "1", "n2": "0", "ac": 1},
    {"name": "R1", "type": "R", "n1": "1", "n2": "2", "value": "1k"},
    {"name": "C1", "type": "C", "n1": "2", "n2": "0", "value": "0.1u"},
    {"name": "PV1", "type": "VM", "n1": "2", "n2": "0", "r": "1M"}],
    "analyses": [{"type": "ac", "f_start": 10, "f_stop": 100000}]}
TRAN = {"title": "RC-цепь", "elements": [
    {"name": "E1", "type": "V", "n1": "1", "n2": "0", "value": 5, "wave": "step"},
    {"name": "R1", "type": "R", "n1": "1", "n2": "2", "value": "10k"},
    {"name": "C1", "type": "C", "n1": "2", "n2": "0", "value": "10u"},
    {"name": "PV1", "type": "VM", "n1": "2", "n2": "0", "r": "1M"}],
    "analyses": [{"type": "tran", "t_stop": 0.5}]}


def circuits():
    return load_circuit(DC), load_circuit(AC), load_circuit(TRAN)
