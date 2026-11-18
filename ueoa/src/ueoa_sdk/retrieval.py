"""Lexical retrieval over processed documents: paragraph-aware chunks with absolute offsets, BM25 search over several
queries, neighbour expansion, verbatim stitching of adjacent chunks and verbatim quote location.

A processed document is `{document_id, status?, sections: [{section_id, title, start, end, text, page?}]}`, where
`start` is the absolute offset of the section text in the original document.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = ["CHUNK_CHARS", "Chunk", "Index", "Span", "chunk", "locate", "source_text", "tokens"]

CHUNK_CHARS = 800
_TOKEN = re.compile(r"[^\W_]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "shall",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "der",
        "die",
        "das",
        "und",
        "ist",
        "von",
        "zu",
        "mit",
        "den",
        "dem",
        "ein",
        "eine",
        "im",
        "wie",
        "wer",
    ]
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    section_id: str
    section: str  # section title
    page: int | None
    start: int  # absolute offsets in the document text
    end: int
    text: str
    position: int  # index in the corpus; consecutive positions in one section are adjacent text


@dataclass(frozen=True)
class Span:
    document_id: str
    section: str
    page: int | None
    start: int
    end: int
    text: str
    chunks: tuple[Chunk, ...]


def locate(quote: str, text: str) -> tuple[int, int] | None:
    """Span of `quote` in `text`: exact match first, then a whitespace-insensitive match."""
    index = text.find(quote)
    if quote and index >= 0:
        return index, index + len(quote)
    words = quote.split()
    if not words:
        return None
    match = re.search(r"\s+".join(map(re.escape, words)), text)
    return match.span() if match else None


def _sections(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in document.get("sections") or [] if isinstance(s, dict)]  # pyright: ignore[reportUnknownVariableType]


def source_text(document: dict[str, Any]) -> str:
    """The document text rebuilt from its sections; gaps (headings) become spaces so absolute offsets hold."""
    parts: list[str] = []
    cursor = 0
    for section in sorted(_sections(document), key=lambda s: int(s.get("start", 0))):
        start, text = int(section.get("start", 0)), str(section.get("text", ""))
        if start < cursor:
            continue
        parts += [" " * (start - cursor), text]
        cursor = start + len(text)
    return "".join(parts)


def _paragraphs(text: str, limit: int) -> list[tuple[int, int]]:
    """Trimmed paragraph spans; paragraphs longer than `limit` are cut at sentence or word boundaries."""
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\s*\n\s*\n|\s*\Z)", text, re.DOTALL):
        start, end = match.span()
        while end - start > limit:
            cut = text.rfind(". ", start, start + limit) + 1
            if cut <= start + limit // 2:
                cut = text.rfind(" ", start, start + limit)
            if cut <= start:
                cut = start + limit
            spans.append((start, start + len(text[start:cut].rstrip())))
            start = cut
            while text[start].isspace():
                start += 1
        spans.append((start, end))
    return spans


def _page(section: dict[str, Any]) -> int | None:
    page = section.get("page")
    return page if isinstance(page, int) and not isinstance(page, bool) else None


def chunk(documents: Iterable[dict[str, Any]], limit: int = CHUNK_CHARS) -> list[Chunk]:
    """~`limit`-char chunks of whole paragraphs, never across sections; unreadable documents are skipped."""
    chunks: list[Chunk] = []
    for document in documents:
        if document.get("status", "ok") != "ok":
            continue
        for section in _sections(document):
            text, base = str(section.get("text", "")), int(section.get("start", 0))
            groups: list[tuple[int, int]] = []
            for start, end in _paragraphs(text, limit):
                if groups and end - groups[-1][0] <= limit:
                    groups[-1] = (groups[-1][0], end)
                else:
                    groups.append((start, end))
            for start, end in groups:
                chunks.append(
                    Chunk(
                        chunk_id=f"c{len(chunks) + 1}",
                        document_id=str(document.get("document_id")),
                        section_id=str(section.get("section_id", "")),
                        section=str(section.get("title", "")),
                        page=_page(section),
                        start=base + start,
                        end=base + end,
                        text=text[start:end],
                        position=len(chunks),
                    )
                )
    return chunks


def tokens(text: str) -> list[str]:
    """Lower-cased words without stopwords, crudely stemmed (one suffix off, then 7-char prefix)."""
    stems: list[str] = []
    for word in _TOKEN.findall(text.lower()):
        if word in _STOPWORDS:
            continue
        for suffix in ("ing", "ed", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        stems.append(word[:7])
    return stems


class Index:
    """BM25 index over the chunks of some processed documents."""

    def __init__(
        self, documents: Iterable[dict[str, Any]], *, limit: int = CHUNK_CHARS, k1: float = 1.5, b: float = 0.75
    ):
        documents = [d for d in documents if isinstance(d, dict)]  # pyright: ignore[reportUnnecessaryIsInstance]
        self.chunks = chunk(documents, limit)
        self._texts = {
            (str(d.get("document_id")), str(s.get("section_id", ""))): (int(s.get("start", 0)), str(s.get("text", "")))
            for d in documents
            for s in _sections(d)
        }
        self._docs = [Counter(tokens(c.text)) for c in self.chunks]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._average = (sum(self._lengths) / len(self._docs) if self._docs else 0.0) or 1.0
        self._k1, self._b = k1, b

    def scores(self, query: str) -> list[float]:
        terms = set(tokens(query))
        count = len(self._docs)
        idf = {t: math.log(1 + (count - n + 0.5) / (n + 0.5)) for t in terms if (n := sum(t in d for d in self._docs))}
        result: list[float] = []
        for doc, length in zip(self._docs, self._lengths, strict=True):
            norm = self._k1 * (1 - self._b + self._b * length / self._average)
            result.append(sum(w * doc[t] * (self._k1 + 1) / (doc[t] + norm) for t, w in idf.items() if t in doc))
        return result

    def search(self, queries: Iterable[str], top_k: int) -> list[tuple[Chunk, float]]:
        """Best `top_k` chunks with a positive score; a chunk scores the max over all queries."""
        best = [0.0] * len(self.chunks)
        for query in queries:
            best = [max(a, s) for a, s in zip(best, self.scores(query), strict=True)]
        ranked = sorted((i for i, s in enumerate(best) if s > 0), key=lambda i: (-best[i], i))[:top_k]
        return [(self.chunks[i], best[i]) for i in ranked]

    def expand(self, hits: Iterable[Chunk], neighbors: int) -> list[Chunk]:
        """The hits plus up to `neighbors` adjacent chunks of the same section on each side, in document order."""
        chosen: set[int] = set()
        for hit in hits:
            for position in range(hit.position - neighbors, hit.position + neighbors + 1):
                if 0 <= position < len(self.chunks) and _same_section(self.chunks[position], hit):
                    chosen.add(position)
        return [self.chunks[i] for i in sorted(chosen)]

    def stitch(self, chunks: Iterable[Chunk]) -> list[Span]:
        """Merge runs of adjacent chunks of one section into verbatim spans of the document text."""
        runs: list[list[Chunk]] = []
        for item in sorted(set(chunks), key=lambda c: c.position):
            last = runs[-1][-1] if runs else None
            if last is not None and item.position == last.position + 1 and _same_section(item, last):
                runs[-1].append(item)
            else:
                runs.append([item])
        spans: list[Span] = []
        for run in runs:
            first, last = run[0], run[-1]
            base, text = self._texts[(first.document_id, first.section_id)]
            body = text[first.start - base : last.end - base]
            spans.append(Span(first.document_id, first.section, first.page, first.start, last.end, body, tuple(run)))
        return spans


def _same_section(a: Chunk, b: Chunk) -> bool:
    return (a.document_id, a.section_id) == (b.document_id, b.section_id)
