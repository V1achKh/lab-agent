#!/usr/bin/env bash
# Запуск моделей через llama.cpp (без Ollama): сервер эмбеддингов + сервер чат-модели.
#
#   scripts/llamacpp.sh gemma12          # Gemma 4 12B (≈7,4 ГБ, понимает картинки) — для тестов
#   scripts/llamacpp.sh qwen27           # Qwen3.8-27B в 3 битах (≈13,1 ГБ) — основная
#   (у qwen27 «зрение» (mmproj) работает на процессоре, чтобы не занимать видеопамять;
#    MMPROJ_CPU=0 — перенести его на видеокарту, MMPROJ_CPU=1 — то же для gemma12)
#
# Где берутся модели:
#   если в папке models/<имя>/ лежит .gguf — используется он (скачанный вручную);
#   иначе модель скачивается с Hugging Face сама (в ~/.cache/llama.cpp).
#     models/gemma12/   gemma-4-12b-it-UD-Q4_K_XL.gguf  (+ mmproj-*.gguf для картинок)
#     models/qwen27/    Qwen3.8-27B-UD-Q3_K_XL.gguf     (+ mmproj-*.gguf для картинок)
#     models/bge-m3/    bge-m3-q8_0.gguf (из ggml-org/bge-m3-Q8_0-GGUF)
#
# Любые дополнительные аргументы передаются llama-server чат-модели.
# Переменные: CTX (контекст), UBATCH (размер пакета; влияет на число токенов картинки), CHAT_PORT (8080), EMBED_PORT (8081).
# Ctrl+C останавливает обе модели. Логи — в logs/.
set -euo pipefail
cd "$(dirname "$0")/.."

PRESET="${1:-gemma12}"
shift || true
CHAT_PORT="${CHAT_PORT:-8080}"
EMBED_PORT="${EMBED_PORT:-8081}"

if ! command -v llama-server >/dev/null; then
    echo "llama-server не найден. Установите llama.cpp (см. INSTALL.md, шаг 3)."
    exit 1
fi

case "$PRESET" in
    gemma12)
        HF="unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL"; ALIAS="gemma4-12b"; CTX="${CTX:-32768}"
        UBATCH="${UBATCH:-2048}" ;;
    qwen27)
        HF="unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL";    ALIAS="qwen3.8-27b"; CTX="${CTX:-16384}"
        MMPROJ_CPU="${MMPROJ_CPU:-1}"; UBATCH="${UBATCH:-1024}" ;;
    *)
        echo "Неизвестная модель «$PRESET». Варианты: gemma12, qwen27"; exit 1 ;;
esac

# Найти локальную модель в models/<папка>: основной .gguf и (необязательно) mmproj.
# Результат — аргументы для llama-server в массиве SRC.
model_source() {
    local dir="models/$1" hf="$2" main="" proj=""
    if [ -d "$dir" ]; then
        main=$(find "$dir" -maxdepth 1 -name '*.gguf' ! -iname '*mmproj*' | sort | head -n 1)
        proj=$(find "$dir" -maxdepth 1 -iname '*mmproj*.gguf' | sort | head -n 1)
    fi
    if [ -n "$main" ]; then
        SRC=(-m "$main")
        if [ -n "$proj" ]; then
            SRC+=(--mmproj "$proj")
            # «Зрение» на процессоре: картинка обрабатывается на пару секунд дольше,
            # зато ~0,9 ГБ видеопамяти остаются модели. Для Qwen 27B включено по умолчанию.
            [ "${MMPROJ_CPU:-0}" = "1" ] && SRC+=(--no-mmproj-offload)
        fi
        SRC_INFO="локальный файл $main${proj:+ + $proj}"
        if [ -z "$proj" ] && [ "$1" != "bge-m3" ]; then
            SRC_INFO="$SRC_INFO (без mmproj — модель не будет видеть картинки, используйте --text-only)"
        fi
    else
        SRC=(-hf "$hf")
        SRC_INFO="$hf (скачается с Hugging Face, если ещё нет в ~/.cache/llama.cpp)"
    fi
}

mkdir -p logs

# 1. Эмбеддинги (bge-m3) — на процессоре (-ngl 0), чтобы не занимать видеопамять.
EMB_PID=""
if curl -sf "http://127.0.0.1:${EMBED_PORT}/health" >/dev/null 2>&1; then
    echo "Сервер эмбеддингов уже запущен на порту ${EMBED_PORT}."
else
    model_source bge-m3 "ggml-org/bge-m3-Q8_0-GGUF:Q8_0"
    echo "Эмбеддинги bge-m3: ${SRC_INFO}"
    echo "Запускаю на порту ${EMBED_PORT} (лог: logs/embed.log)…"
    llama-server "${SRC[@]}" --alias bge-m3 \
        --embeddings --pooling cls -ngl 0 -c 8192 -b 8192 -ub 8192 -np 1 \
        --host 127.0.0.1 --port "${EMBED_PORT}" > logs/embed.log 2>&1 &
    EMB_PID=$!
fi
cleanup() { [ -n "$EMB_PID" ] && kill "$EMB_PID" 2>/dev/null || true; }
trap cleanup EXIT

# 2. Чат-модель — на видеокарте. -np 1: весь контекст одному агенту (иначе он делится на «слоты»).
#    --jinja обязателен: без него не работают вызовы инструментов.
model_source "$PRESET" "$HF"
echo
echo "Чат-модель ${ALIAS}: ${SRC_INFO}"
echo "Порт ${CHAT_PORT}, контекст ${CTX}. Остановить: Ctrl+C."
echo "В .env: LAB_LLM_MODEL=${ALIAS}"
echo
# -b/-ub: размер пакета. Картинка со схемой кодируется до ~1100 токенов; при -ub 512 (по умолчанию)
# llama.cpp урезает её до 512 и мелкие подписи на схеме читаются хуже. UBATCH=512 вернёт экономию памяти.
llama-server "${SRC[@]}" --alias "$ALIAS" --jinja -ngl 99 -c "$CTX" -np 1 -b "$UBATCH" -ub "$UBATCH" \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --host 127.0.0.1 --port "${CHAT_PORT}" "$@"
