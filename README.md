# LabAgent

Локальный ИИ-агент для лабораторных по электротехнике (ИУ5-34Б). По картинке варианта и
методичке PDF делает отчёт `ИУ5-34Б Фамилия Лаб.1.docx` / `.pdf`.

Gemma 4 12B (llama.cpp) + RAG (bge-m3) + свой MCP-сервер + ngspice.

Установка — [INSTALL.md](INSTALL.md), как устроено — [ОПИСАНИЕ_ПРОЕКТА.md](ОПИСАНИЕ_ПРОЕКТА.md).

## Файлы, которых нет в репозитории

Модели, методички и варианты в git не лежат. Положить самим:
- `models/gemma12/`, `models/bge-m3/` — файлы .gguf (см. INSTALL.md);
- `tasks/lab1/` — методичка PDF и картинки вариантов;
- `data/docs/` — свои PDF для RAG (ГОСТ, «Типичные ошибки»), потом `python -m labagent index`.

## Запуск

Терминал 1:
```bash
cd ~/lab-agent && scripts/llamacpp.sh
```

Терминал 2:
```bash
cd ~/lab-agent && source .venv/bin/activate.fish
python -m labagent agent tasks/lab1/ --variant 50     # агент через MCP
python -m labagent pipeline tasks/lab1/ --variant 50  # шаги задаёт код
python -m labagent ask "какие поля по ГОСТ 7.32?"     # вопрос к RAG (--no-rag для сравнения)
```

Результат — в `runs/<дата>-<название>/`. Сверяйте `schematic.png` с заданием.

## Команды

| Команда | Что делает |
|---|---|
| `index` | проиндексировать `data/docs/` |
| `ask`, `search` | вопрос / поиск по базе |
| `pipeline`, `agent [--universal]` | выполнить лабу |
| `review отчёт.docx` | проверить отчёт |
| `compare` | сравнить режимы |
| `check` | проверить установку |

## Если не работает

| Симптом | Что делать |
|---|---|
| «никто не слушает порт 8080» | не запущен `scripts/llamacpp.sh` |
| `No module named …` | `source .venv/bin/activate.fish` |
| `индекс не построен` | `python -m labagent index` |
| нет PDF | `sudo pacman -S libreoffice-fresh` |
