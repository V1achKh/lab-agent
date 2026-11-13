"""RAG: поиск по методичкам и ГОСТам.

Индексация (один раз, `python -m labagent index`):
    документы → куски по заголовкам (~900 символов) → эмбеддинги (bge-m3) → data/index/
Поиск:
    вопрос → эмбеддинг → косинусная близость с кусками   (смысловой поиск)
           + BM25 по словам                              (точные термины: «7.32», «PV1»)
    → объединение двух списков методом RRF → top-k кусков в промпт модели.

Векторная «база» — это просто матрица numpy в файле: для сотни кусков
отдельный сервер БД не нужен, а принцип ровно тот же, что у Chroma/Qdrant.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .llm import LLM

CHUNK = 900
OVERLAP = 150


@dataclass
class Chunk:
    text: str
    source: str
    heading: str

    def as_dict(self):
        return {"text": self.text, "source": self.source, "heading": self.heading}


# ---------------------------------------------------------------- чтение документов

def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        from docx import Document

        out = []
        for p in Document(str(path)).paragraphs:
            if not p.text.strip():
                continue
            style = (p.style.name or "").lower()
            out.append(("## " if style.startswith("heading") or style.startswith("заголовок") else "") + p.text)
        return "\n\n".join(out)
    return ""


def split_markdown(text: str, source: str) -> list[Chunk]:
    """Режем по заголовкам, длинные разделы — по абзацам с перекрытием."""
    chunks: list[Chunk] = []
    heading_stack: list[str] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        head = " / ".join(heading_stack)
        paras = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        cur = ""
        for p in paras:
            if len(cur) + len(p) > CHUNK and cur:
                chunks.append(Chunk(cur.strip(), source, head))
                cur = cur[-OVERLAP:] + "\n" + p
            else:
                cur = (cur + "\n\n" + p) if cur else p
            while len(cur) > CHUNK * 1.5:  # очень длинный абзац
                chunks.append(Chunk(cur[:CHUNK].strip(), source, head))
                cur = cur[CHUNK - OVERLAP:]
        if cur.strip():
            chunks.append(Chunk(cur.strip(), source, head))

    for line in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            heading_stack[:] = heading_stack[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return chunks


def load_chunks(docs_dir: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".md", ".txt", ".pdf", ".docx"):
            text = read_document(path)
            if text.strip():
                chunks += split_markdown(text, path.name)
    return chunks


# ---------------------------------------------------------------- BM25

_WORD = re.compile(r"[\wё]+", re.I)


def tokenize(text: str) -> list[str]:
    """Нижний регистр, ё→е и грубая «основа» слова (первые 6 букв):
    «резистора», «резистором» → «резист». Для BM25 по-русски этого хватает."""
    words = _WORD.findall(text.lower().replace("ё", "е"))
    return [w if w.replace(".", "").isdigit() else w[:6] for w in words]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df = Counter(w for d in self.docs for w in d)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            s = 0.0
            for w in query:
                if w in d:
                    tf = d[w]
                    s += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            out[i] = s
        return out


# ---------------------------------------------------------------- индекс

class RagIndex:
    def __init__(self, chunks: list[Chunk], emb: np.ndarray | None, model: str):
        self.chunks = chunks
        self.emb = emb
        self.model = model
        self.bm25 = BM25([tokenize(c.heading + " " + c.text) for c in chunks])

    # ---- построение и загрузка
    @classmethod
    def build(cls, llm: LLM, docs_dir: Path | None = None, index_dir: Path | None = None) -> "RagIndex":
        docs_dir = Path(docs_dir or config.DOCS_DIR)
        index_dir = Path(index_dir or config.INDEX_DIR)
        chunks = load_chunks(docs_dir)
        if not chunks:
            raise RuntimeError(f"в {docs_dir} нет документов (.md .txt .pdf .docx)")
        # в эмбеддинг кладём и заголовок раздела: так кусок «знает», о чём он
        emb = llm.embed([f"{c.heading}\n{c.text}" for c in chunks])
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", emb)
        (index_dir / "chunks.json").write_text(json.dumps(
            {"model": llm.embed_model, "chunks": [c.as_dict() for c in chunks]}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        return cls(chunks, emb, llm.embed_model)

    @classmethod
    def merge(cls, *indexes: "RagIndex | None") -> "RagIndex | None":
        """Постоянная база (ГОСТ, замечания) + методичка этого задания → один индекс,
        общий рейтинг кусков. Эмбеддинги должны быть от одной модели."""
        parts = [i for i in indexes if i is not None]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        chunks = [c for i in parts for c in i.chunks]
        embs = [i.emb for i in parts]
        emb = np.vstack(embs) if all(e is not None for e in embs) and len({e.shape[1] for e in embs}) == 1 else None
        return cls(chunks, emb, parts[0].model)

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "RagIndex":
        index_dir = Path(index_dir or config.INDEX_DIR)
        meta_path = index_dir / "chunks.json"
        if not meta_path.exists():
            raise RuntimeError("индекс не построен: запустите `python -m labagent index`")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in meta["chunks"]]
        emb = np.load(index_dir / "embeddings.npy")
        return cls(chunks, emb, meta.get("model", ""))

    # ---- поиск
    def search(self, query: str, llm: LLM | None = None, k: int = 4, mode: str = "hybrid") -> list[dict]:
        """mode: hybrid (по умолчанию) | vector | bm25."""
        rankings = []
        vec_scores = None
        if mode in ("hybrid", "vector") and self.emb is not None and llm is not None:
            q = llm.embed([query])[0]
            vec_scores = self.emb @ q
            rankings.append(np.argsort(-vec_scores))
        if mode in ("hybrid", "bm25") or not rankings:
            bm = self.bm25.scores(tokenize(query))
            rankings.append(np.argsort(-bm))
        # Reciprocal Rank Fusion: складываем 1/(60 + место) из каждого списка
        fused = Counter()
        for ranking in rankings:
            for place, idx in enumerate(ranking[:50]):
                fused[int(idx)] += 1.0 / (60 + place)
        out = []
        for idx, score in fused.most_common(k):
            c = self.chunks[idx]
            out.append({**c.as_dict(), "score": round(score, 5),
                        "cosine": None if vec_scores is None else round(float(vec_scores[idx]), 4)})
        return out


def format_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['source']} — {r['heading']}\n{r['text']}")
    return "\n\n".join(parts)
