"""Клиент для OpenAI-совместимого API (Ollama, llama-server, LM Studio, vLLM).

Написан на стандартной библиотеке, без пакета openai — так видно, что
«общение с моделью» — это просто POST-запрос с JSON:
    POST /v1/chat/completions  {"model", "messages", "tools"?}
    POST /v1/embeddings        {"model", "input"}
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

import numpy as np

from . import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class LLMError(RuntimeError):
    pass


def _post(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    url = base_url.rstrip("/") + path
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.LLM_API_KEY}",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:800]
        raise LLMError(f"{url}: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(
            f"не удалось подключиться к {url} ({exc.reason}). "
            "Запущен ли сервер модели (ollama serve / llama-server)?") from exc


def strip_thinking(text: str | None) -> str:
    """«Думающие» модели иногда возвращают <think>...</think> прямо в ответе."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # открывающий тег мог «съесть» шаблон чата
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str):
    """Достаёт первый JSON-объект из ответа (модели любят оборачивать его в ```json ... ```)."""
    text = strip_thinking(text)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("в ответе модели нет JSON-объекта")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("JSON в ответе модели оборван")


def user_message(text: str, images: str | list[str] | None = None) -> dict:
    """Сообщение пользователя; если есть картинки (схема, данные варианта) — в формате OpenAI vision.
    Нужна модель, которая понимает изображения (Gemma 4; в llama-server — с --mmproj)."""
    if isinstance(images, (str, bytes)) or hasattr(images, "__fspath__"):
        images = [images]
    images = [str(i) for i in (images or []) if i]
    if not images:
        return {"role": "user", "content": text}
    import base64
    import mimetypes
    from pathlib import Path

    content = [{"type": "text", "text": text}]
    for image in images:
        mime = mimetypes.guess_type(image)[0] or "image/png"
        data = base64.b64encode(Path(image).read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    return {"role": "user", "content": content}


class LLM:
    def __init__(self, url: str | None = None, model: str | None = None,
                 embed_url: str | None = None, embed_model: str | None = None):
        self.url = url or config.LLM_URL
        self.model = model or config.LLM_MODEL
        self.embed_url = embed_url or config.EMBED_URL
        self.embed_model = embed_model or config.EMBED_MODEL
        self.calls = 0          # статистика для сравнения режимов
        self.seconds = 0.0
        self.tokens = 0

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False, temperature: float | None = None,
             max_tokens: int | None = None) -> dict:
        """Возвращает сообщение ассистента: {"role", "content", "tool_calls"?}."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
            **config.LLM_EXTRA,
        }
        payload["max_tokens"] = max_tokens or config.LLM_MAX_TOKENS
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        t0 = time.time()
        try:
            data = _post(self.url, "/chat/completions", payload, config.LLM_TIMEOUT)
        except LLMError as exc:
            # не все серверы понимают response_format — пробуем без него
            if json_mode and "HTTP 4" in str(exc):
                payload.pop("response_format", None)
                data = _post(self.url, "/chat/completions", payload, config.LLM_TIMEOUT)
            else:
                raise
        self.calls += 1
        self.seconds += time.time() - t0
        self.tokens += int((data.get("usage") or {}).get("total_tokens") or 0)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"неожиданный ответ сервера: {str(data)[:500]}") from exc
        msg["content"] = strip_thinking(msg.get("content"))
        # некоторые серверы кладут вызов инструмента в текст вместо tool_calls
        if tools and not msg.get("tool_calls"):
            msg["tool_calls"] = _tool_calls_from_text(msg["content"], tools)
            if msg["tool_calls"]:
                msg["content"] = ""
        if not msg.get("tool_calls"):
            msg.pop("tool_calls", None)
        return msg

    def ask(self, system: str, user: str, **kw) -> str:
        return self.chat([{"role": "system", "content": system},
                          {"role": "user", "content": user}], **kw)["content"]

    def embed(self, texts: list[str], batch: int = 16) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            data = _post(self.embed_url, "/embeddings", {"model": self.embed_model, "input": chunk},
                         config.LLM_TIMEOUT)
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            vecs += [d["embedding"] for d in items]
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)


_TOOL_TEXT_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _tool_calls_from_text(text: str, tools: list[dict]) -> list[dict]:
    """Запасной разбор формата Qwen/Hermes: <tool_call>{"name":..., "arguments":...}</tool_call>."""
    names = {t["function"]["name"] for t in tools}
    calls = []
    for k, m in enumerate(_TOOL_TEXT_RE.finditer(text or "")):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if obj.get("name") in names:
            args = obj.get("arguments", {})
            calls.append({"id": f"call_text_{k}", "type": "function",
                          "function": {"name": obj["name"],
                                       "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}})
    return calls
