"""PDF -> sections -> chunks.

Design choices (each one is something to measure, not assume):
- Section-aware: a chunk never crosses a section heading, so a citation's section is exact.
- Page-accurate: every chunk records the page its first word came from.
- References are dropped: they match many queries lexically and answer none of them.
- Running headers/footers repeated across pages are removed before chunking.
- Chunk size is in words (a stable, tokenizer-free proxy); size and overlap come from Settings.

Run: papertrail-ingest --pdf-dir data/corpus/pdf --out data/processed/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from papertrail.schemas import Chunk

HEADING_NUMBERED = re.compile(
    r"^(?:\d+(?:\.\d+){0,2}\.?|[A-H](?:\.\d+){0,2}\.?)\s+[A-Z][^.!?]{1,90}$"
)
HEADING_NAMED = re.compile(
    r"^(abstract|introduction|related work|background|conclusions?|discussion|"
    r"acknowledge?ments?|references|bibliography|appendix(?:\s+[a-z])?)$",
    re.IGNORECASE,
)
STOP_SECTIONS = re.compile(r"^(references|bibliography)$", re.IGNORECASE)
APPENDIX_START = re.compile(r"^(appendix\b|[A-H](?:\.\d+)*\.?\s+[A-Z])", re.IGNORECASE)


def normalize(text: str) -> str:
    """Canonical text used for both stored chunks and evidence matching.

    Joins words hyphenated across line breaks, unifies ligatures and quotes, collapses whitespace.
    """
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"')
    text = text.replace("”", '"').replace("–", "-").replace("—", "-")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "body"


@dataclass
class Line:
    text: str
    page: int
    size: float
    bold: bool
    in_margin: bool = False  # in the top or bottom 8% of the page, where headers/footers live


MARGIN = 0.08


@dataclass
class Section:
    title: str
    lines: list[Line] = field(default_factory=list)


def read_lines(pdf_path: Path) -> list[Line]:
    import pymupdf  # imported lazily so the API image does not need it

    lines: list[Line] = []
    with pymupdf.open(pdf_path) as doc:  # type: ignore[no-untyped-call]
        for pageno, page in enumerate(doc, start=1):
            height = page.rect.height
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    size = max(s["size"] for s in spans)
                    bold = all(s["flags"] & 16 or "Bold" in s["font"] for s in spans)
                    y0, y1 = line["bbox"][1], line["bbox"][3]
                    margin = y1 < height * MARGIN or y0 > height * (1 - MARGIN)
                    lines.append(Line(text, pageno, size, bold, in_margin=margin))
    return lines


def drop_running_lines(lines: list[Line], n_pages: int) -> list[Line]:
    """Remove headers/footers: margin lines that recur (digits ignored) on many pages.

    Only lines in the top/bottom margin are candidates, so a sentence repeated in the body
    (or a recurring table row) is never mistaken for a header.
    """

    def key(ln: Line) -> str:
        return re.sub(r"\d+", "#", ln.text.lower())

    pages_per_key: dict[str, set[int]] = {}
    for ln in lines:
        if ln.in_margin:
            pages_per_key.setdefault(key(ln), set()).add(ln.page)
    needed = max(2, n_pages // 2)
    running = {k for k, pages in pages_per_key.items() if len(pages) >= needed}

    def keep(ln: Line) -> bool:
        if not ln.in_margin:
            return True
        return key(ln) not in running and not re.fullmatch(r"\d{1,3}", ln.text)

    return [ln for ln in lines if keep(ln)]


def body_size(lines: Iterable[Line]) -> float:
    weights: Counter[float] = Counter()
    for ln in lines:
        weights[round(ln.size, 1)] += len(ln.text)
    return weights.most_common(1)[0][0] if weights else 10.0


def is_heading(ln: Line, body: float) -> bool:
    text = ln.text.strip()
    if len(text.split()) > 12 or len(text) < 3:
        return False
    emphasized = ln.bold or ln.size >= body + 0.8
    if not emphasized:
        return False
    return bool(HEADING_NUMBERED.match(text) or HEADING_NAMED.match(text))


def split_sections(lines: list[Line]) -> list[Section]:
    body = body_size(lines)
    sections = [Section(title="Front matter")]
    skipping_refs = False
    for ln in lines:
        if is_heading(ln, body):
            title = ln.text.strip()
            if STOP_SECTIONS.match(title):
                skipping_refs = True
                continue
            if skipping_refs and not APPENDIX_START.match(title):
                continue
            skipping_refs = False
            sections.append(Section(title=title))
            continue
        if not skipping_refs:
            sections[-1].lines.append(ln)
    return [s for s in sections if s.lines]


def section_words(section: Section) -> list[tuple[str, int]]:
    """(word, page) pairs for a section, with words split by a line-end hyphen re-joined."""
    words: list[tuple[str, int]] = []
    hyphen_open = False
    for ln in section.lines:
        parts = normalize(ln.text).split()
        if not parts:
            continue
        if hyphen_open and words and parts[0][:1].islower():
            prev, page = words.pop()
            words.append((prev[:-1] + parts[0], page))
            parts = parts[1:]
        words.extend((w, ln.page) for w in parts)
        hyphen_open = bool(words) and words[-1][0].endswith("-") and len(words[-1][0]) > 1
    return words


def chunk_section(
    paper_id: str, section: Section, index: int, size: int, overlap: int
) -> Iterator[Chunk]:
    words = section_words(section)
    step = max(1, size - overlap)
    starts = range(0, max(1, len(words) - overlap), step) if len(words) > size else [0]
    for n, start in enumerate(starts):
        piece = words[start : start + size]
        if not piece:
            break
        yield Chunk(
            chunk_id=f"{paper_id}:{index:02d}-{slug(section.title)}:{n:03d}",
            paper_id=paper_id,
            section=section.title,
            page=piece[0][1],
            text=" ".join(w for w, _ in piece),
        )


def ingest_pdf(pdf_path: Path, paper_id: str, size: int, overlap: int) -> list[Chunk]:
    lines = read_lines(pdf_path)
    n_pages = max((ln.page for ln in lines), default=0)
    lines = drop_running_lines(lines, n_pages)
    chunks: list[Chunk] = []
    for i, section in enumerate(split_sections(lines)):
        chunks.extend(chunk_section(paper_id, section, i, size, overlap))
    return chunks


def main(argv: list[str] | None = None) -> int:
    from papertrail.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(prog="papertrail-ingest")
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/corpus/pdf"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--chunk-words", type=int, default=settings.chunk_words)
    parser.add_argument("--overlap-words", type=int, default=settings.chunk_overlap_words)
    args = parser.parse_args(argv)

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs in {args.pdf_dir}; run scripts/download_corpus.py first")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for pdf in pdfs:
            chunks = ingest_pdf(pdf, pdf.stem, args.chunk_words, args.overlap_words)
            sections = len({c.section for c in chunks})
            print(f"{pdf.stem:28s} {sections:3d} sections {len(chunks):4d} chunks")
            for c in chunks:
                fh.write(c.model_dump_json() + "\n")
            total += len(chunks)
    print(f"\n{total} chunks from {len(pdfs)} papers -> {args.out}")
    return 0


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as fh:
        return [Chunk.model_validate(json.loads(line)) for line in fh if line.strip()]


if __name__ == "__main__":
    sys.exit(main())
