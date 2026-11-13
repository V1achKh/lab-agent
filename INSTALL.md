# Полная инструкция: от установки до результата

Garuda / Arch Linux, RTX 5060 Ti 16 ГБ. Всё через llama.cpp, без Ollama.
Команды для fish (оболочка Garuda по умолчанию); отличия для bash указаны.

Что получится в конце:
- отчёты по лабораторной №1 в `runs/…/` — файл называется как требует преподаватель: `ИУ5-34Б Фамилия Лаб.1.docx` и `.pdf`;
- таблица сравнения режимов и моделей в `results/compare-*.md` — главный результат для защиты.

---

## Шаг 1. Драйвер NVIDIA

```bash
nvidia-smi
```

Должна показаться RTX 5060 Ti и версия CUDA справа вверху. В Garuda драйвер обычно уже стоит.
Если команды нет — установите драйвер через Garuda Settings Manager → Hardware Configuration
(или `sudo pacman -S nvidia-open-dkms nvidia-utils`) и перезагрузитесь.

## Шаг 2. Системные пакеты

```bash
sudo pacman -S --needed python git curl ngspice libreoffice-fresh ttf-liberation podman poppler
ngspice --version          # симулятор цепей
soffice --version          # LibreOffice — делает PDF
```

- `ngspice` — расчёт цепей;
- `libreoffice-fresh` — конвертация отчёта в PDF;
- `ttf-liberation` — замена Times New Roman (для настоящего TNR: `paru -S ttf-ms-fonts`);
- `podman` — песочница для универсального режима;
- `poppler` — страницы PDF-задания как картинки для модели (запасной вариант к pypdfium2 из pip).

## Шаг 3. llama.cpp с CUDA

```bash
paru -S llama.cpp-cuda     # собирается из исходников, 10–30 минут
llama-server --version     # в выводе должна быть CUDA и ваша видеокарта
```

<details><summary>Если пакет из AUR не собирается — сборка вручную</summary>

```bash
sudo pacman -S --needed cuda cmake base-devel
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j 6
fish_add_path ~/llama.cpp/build/bin      # bash: echo 'export PATH=~/llama.cpp/build/bin:$PATH' >> ~/.bashrc
```

Ошибка «unsupported GNU version» — посмотрите `ls /usr/bin/g++-*` и добавьте к первой команде
cmake `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14` (своя версия), удалите `build/` и соберите заново.
</details>

## Шаг 4. Проект

```bash
cd ~
unzip lab-agent.zip
cd lab-agent
python -m venv .venv
source .venv/bin/activate.fish           # bash: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Теперь в папке проекта есть файл `.env` с настройками. Он скрытый (имя начинается с точки):
в файловом менеджере Dolphin его видно после Ctrl+H. Откройте его в редакторе:

```bash
kate .env        # графический редактор (в Garuda KDE он есть); или в терминале: nano .env
```

Всё уже настроено под llama.cpp. Поменять нужно только блок «Титульный лист» в самом низу —
впишите свои фамилию и группу:

```
LAB_STUDENT=Иванов И. И.
LAB_GROUP=ИУ5-34Б
```

Сохраните (Ctrl+S) и закройте. Когда позже перейдёте на Qwen3.8-27B (шаг 12), в этом же файле
поменяйте `LAB_LLM_MODEL=gemma4-12b` на `LAB_LLM_MODEL=qwen3.8-27b`.

`source .venv/bin/activate.fish` нужно повторять в каждом новом терминале перед командами проекта.

## Шаг 5. Проверка без модели

```bash
python tests/test_core.py        # все строки OK
python -m labagent demo          # отчёты по эталонным цепям из tasks/
```

Откройте PDF в `runs/` — так выглядит оформление (схема, описание, уравнения, погрешности).
Модель здесь ещё не участвует: это проверка, что ngspice, рисование и LibreOffice работают.

## Шаг 6. Модели (квантованные)

**Терминал 1** — оставьте его открытым всё время работы:

```bash
cd ~/lab-agent
scripts/llamacpp.sh gemma12
```

Скрипт поднимает два сервера. Модели он берёт так:
- если вы **скачали .gguf вручную** — положите их в папку проекта `models/`, каждую в свою
  подпапку: `models/gemma12/`, `models/qwen27/`, `models/bge-m3/` (подробно — `models/README.md`);
- если подпапка пустая — скрипт сам скачает модель с Hugging Face (в `~/.cache/llama.cpp`).

В выводе скрипта видно, что он использует: «локальный файл …» или «скачается с Hugging Face».

| Модель | Квант | Размер | Где работает | Порт |
|---|---|---|---|---|
| bge-m3 (эмбеддинги для RAG) | Q8_0 (ggml-org) | 0,64 ГБ | процессор | 8081 |
| Gemma 4 12B (чат, понимает картинки) | UD-Q4_K_XL | 7,4 ГБ | видеокарта | 8080 |

Готово, когда в выводе появилось `server is listening on http://127.0.0.1:8080`.

## Шаг 7. Проверка связки

**Терминал 2:**

```bash
cd ~/lab-agent
source .venv/bin/activate.fish
python -m labagent check --tools
```

Нужно увидеть `[OK]` у строк: ngspice, LibreOffice, сервер модели, чат, вызов инструментов,
эмбеддинги. `[--]` у песочницы пока нормально (шаг 11).

## Шаг 8. RAG — база знаний

База из двух частей. **Постоянная** — `data/docs/`: ГОСТы, формат отчёта группы и
`zamechaniya_prepodavatelya.md` — за что преподаватель снижает оценку. **Откройте этот файл и
впишите свои пункты** (сейчас там пример). Сюда же положите оригиналы ГОСТов (PDF с текстом).
**Методичка** подаётся к каждому заданию отдельно (`-m`) и в постоянную базу не попадает.

```bash
python -m labagent index
python -m labagent ask "какие поля страницы по ГОСТ 7.32?" --no-rag
python -m labagent ask "какие поля страницы по ГОСТ 7.32?"
python -m labagent ask "что должно быть в отчёте?" -m tasks/lab1/Методичка_Лаб1.pdf
```

`index` — один раз и после каждого изменения в `data/docs/`. Первые две команды `ask` — демонстрация
RAG: без него модель придумывает, с ним отвечает по документам со ссылками [1], [2].

## Шаг 9. Пайплайн — первый отчёт

На вход — только то, что выдал преподаватель: картинка варианта и методичка в PDF. Что есть что,
программа определит сама по содержимому.

```bash
python -m labagent pipeline tasks/lab1/вариант_А.png tasks/lab1/Методичка_Лаб1.pdf
```

Модель смотрит на картинку, читает методичку и составляет описание цепи; всё остальное (расчёт,
схема, отчёт) делает код. В конце — путь к отчёту `ИУ5-34Б Фамилия Лаб.1.docx` и проверка по
списку замечаний преподавателя.

**Обязательно:** откройте `runs/<последняя папка>/schematic.png` и сравните с картинкой задания.
Соединения и номиналы должны совпасть (расположение — нет: схема рисуется своя, как требует
преподаватель). Если не совпадает — модель неверно прочитала картинку, и весь расчёт для другой цепи.

Свои варианты — так же: файлы или папка, и номер варианта, если вариантов в папке/таблице несколько:

```bash
python -m labagent pipeline ~/лабы/лаб1/ --variant 17      # возьмёт картинку с «17» в имени
python -m labagent pipeline ~/лабы/лаб2/ --variant 32      # возьмёт строку 32 из таблицы вариантов
```

## Шаг 10. Агент через MCP

```bash
python -m labagent agent tasks/lab1/ --variant Б
```

Первая строка вывода — `MCP: подключено инструментов — 4: …`: агент сам запустил MCP-сервер.
Дальше видно, какие инструменты модель вызывает и что получает в ответ.

## Шаг 11. Песочница и универсальный агент

```bash
python -m labagent sandbox build     # образ с Python и numpy, ~2–5 минут, нужен интернет
python -m labagent sandbox test      # все шесть проверок изоляции — [OK]
python -m labagent agent tasks/lab1/ --variant 50 --universal
```

В универсальном режиме модель сама пишет код расчёта по методичке; код выполняется в контейнере
без сети и без доступа к вашим файлам. Если `sandbox build` не может скачать образ — допишите в
`.env` строку `LAB_SANDBOX_IMAGE=host` и повторите `sandbox test`.

## Шаг 11а. Любая лабораторная по методичке

Положите в папку методичку и задание варианта (картинку или таблицу вариантов) и запустите:

```bash
python -m labagent agent ~/лабы/лаб3/ --variant 7 --universal
python -m labagent review "runs/<папка>/ИУ5-34Б Фамилия Лаб.3.docx"   # проверить отчёт ещё раз (или свой)
```

Демо на готовом примере (лабы №3 в проекте нет — агент работает только по PDF-методичке):

```bash
python -m labagent agent tasks/lab3/ --universal
```

## Шаг 12. Результат: сравнение

```bash
python -m labagent compare --repeats 5
```

Прогоняет задания из `tasks/*.json` (варианты 50, А, Б лабы 1 и лаба 3) × три режима × 5 раз (на 12B — порядка получаса–часа). Таблица сохраняется
в `results/compare-<дата>.md`: создан ли отчёт, верна ли цепь, верны ли числа, время, число
вызовов модели.

Затем то же на большой модели. В **терминале 1**: Ctrl+C, затем

```bash
scripts/llamacpp.sh qwen27           # Qwen3.8-27B, квант UD-Q3_K_XL, 13,4 ГБ (скачивается один раз)
```

В `.env` замените `LAB_LLM_MODEL=gemma4-12b` на `LAB_LLM_MODEL=qwen3.8-27b`, и в **терминале 2**:

```bash
python -m labagent check --tools
python -m labagent compare --repeats 5
```

Две таблицы (12B и 27B) — это и есть итог работы: как размер модели влияет на надёжность
пайплайна, агента и универсального агента. Индекс RAG пересобирать не нужно — эмбеддинги те же.

## Что сдавать / показывать

- `results/compare-*.md` для обеих моделей;
- 1–2 отчёта из `runs/` (docx и pdf);
- вывод `sandbox test`;
- демонстрация `ask` с RAG и без;
- тезисы для защиты — в README, раздел «Что рассказать на защите».

## Если что-то не так

| Симптом | Что делать |
|---|---|
| `check`: сервер модели не отвечает | в терминале 1 не запущен `scripts/llamacpp.sh` или ещё грузится модель |
| `check`: модели «…» нет в списке | `LAB_LLM_MODEL` в `.env` не совпадает с тем, что печатает скрипт |
| `check`: вызов инструментов `[!!]` | обновите llama.cpp; скрипт уже передаёт `--jinja` |
| llama-server: `out of memory` | `CTX=8192 scripts/llamacpp.sh qwen27` (зрение у Qwen и так на процессоре); закройте браузер/игры, которые держат видеопамять |
| `unknown argument: --no-mmproj-offload` | старая сборка llama.cpp — обновите её или запускайте `MMPROJ_CPU=0 CTX=8192 scripts/llamacpp.sh qwen27` |
| ошибка про квантованный V-кэш | старая сборка llama.cpp: добавьте `-fa on` в конец команды скрипта или обновите |
| модель не понимает картинку | добавьте `--text-only` к `pipeline`, `agent`, `compare` |
| `индекс не построен` | `python -m labagent index` |
| нет PDF | не установлен `libreoffice-fresh` — отчёт остаётся в .docx |
| песочница не найдена | `sudo pacman -S podman`, затем `python -m labagent sandbox build` |
