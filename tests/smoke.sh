#!/usr/bin/env bash
# Сквозная проверка всего проекта БЕЗ настоящей модели: поднимает фейковый
# OpenAI-совместимый сервер и прогоняет check → index → pipeline → agent (через MCP).
# Индекс строится во временную папку, чтобы не испортить настоящий.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8765}
python tests/fake_llm.py $PORT & FAKE=$!
trap 'kill $FAKE 2>/dev/null' EXIT
sleep 1
export LAB_LLM_URL=http://127.0.0.1:$PORT/v1 LAB_EMBED_URL=http://127.0.0.1:$PORT/v1
export LAB_LLM_MODEL=fake LAB_EMBED_MODEL=fake-embed
export LAB_INDEX_DIR=$(mktemp -d) LAB_RUNS_DIR=$(mktemp -d)
python tests/test_core.py
python -m labagent check --tools
python -m labagent index
# как в жизни: на вход только файлы — картинка варианта и методичка в PDF
python -m labagent pipeline tasks/lab1/вариант_А.png tasks/lab1/Методичка_Лаб1.pdf
python -m labagent agent tasks/lab1/ --variant 50
# выбор строки из таблицы вариантов (лаба 2): проверяем только разбор входных файлов
python -c "from labagent.inputs import load_inputs; t=load_inputs('', ['tasks/lab2/Варианты_Лаб2.pdf'], '32'); assert 'LM2931CS' in t.text, t.text; print('таблица вариантов: OK')"
# универсальный режим — только если есть песочница (код модели без изоляции не запускаем)
if python -m labagent sandbox test; then
    python -m labagent agent tasks/lab1_varB.json --universal
else
    echo "Песочницы нет — универсальный режим пропущен (sudo pacman -S podman; python -m labagent sandbox build)"
fi
echo "Смоук-тест пройден. Результаты: $LAB_RUNS_DIR"
