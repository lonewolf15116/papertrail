from pathlib import Path

import pymupdf
import pytest

from papertrail.eval.evidence import EvidenceIndex, squash
from papertrail.ingest import ingest_pdf, normalize, section_words
from papertrail.schemas import Chunk, Evidence, GoldQuestion

FILLER = "Rematerialization trades extra compute for lower peak memory during training. "


def make_paper(path: Path) -> None:
    """A 4-page fake paper: running header, numbered headings, a hyphen break, refs, appendix."""
    doc = pymupdf.open()
    pages = [
        [("h", "Abstract"), ("b", FILLER * 3), ("h", "1 Introduction"), ("b", FILLER * 6)],
        [
            ("h", "2 Method"),
            ("b", "The evictor picks the tensor with the lowest heuris-"),
            ("b", "tic score and frees it immediately."),
            ("b", FILLER * 3),
        ],
        [
            ("h", "3 Results"),
            ("b", FILLER * 2),
            ("h", "References"),
            ("b", "[1] Chen et al. Training deep nets with sublinear memory cost. 2016."),
        ],
        [
            ("b", "[2] Jain et al. Checkmate. MLSys 2020."),
            ("h", "A Proofs"),
            ("b", "Appendix text about optimal schedules stays in the corpus."),
        ],
    ]
    for content in pages:
        page = doc.new_page()
        page.insert_text((72, 40), "Preprint under review", fontsize=8, fontname="helv")
        y = 80.0
        for kind, text in content:
            if kind == "h":
                page.insert_text((72, y), text, fontsize=13, fontname="hebo")
                y += 24
                continue
            words, line = text.split(), ""
            for w in words:  # wrap body text at ~80 chars so each line is a separate PDF line
                if len(line) + len(w) > 80:
                    page.insert_text((72, y), line, fontsize=10, fontname="helv")
                    y, line = y + 13, ""
                line += w + " "
            page.insert_text((72, y), line, fontsize=10, fontname="helv")
            y += 20
    doc.save(path)


@pytest.fixture
def chunks(tmp_path) -> list[Chunk]:
    pdf = tmp_path / "toy.pdf"
    make_paper(pdf)
    return ingest_pdf(pdf, "toy", size=40, overlap=10)


def test_sections_found_and_references_dropped(chunks):
    sections = list(dict.fromkeys(c.section for c in chunks))
    assert sections == ["Abstract", "1 Introduction", "2 Method", "3 Results", "A Proofs"]
    text = " ".join(c.text for c in chunks)
    assert "Chen et al." not in text and "Checkmate. MLSys" not in text


def test_running_header_removed(chunks):
    assert not any("Preprint under review" in c.text for c in chunks)


def test_pages_and_hyphen_join(chunks):
    method = [c for c in chunks if c.section == "2 Method"]
    assert method[0].page == 2
    assert "heuristic score" in method[0].text
    assert next(c for c in chunks if c.section == "A Proofs").page == 4


def test_chunks_respect_size_and_overlap(chunks):
    intro = [c for c in chunks if c.section == "1 Introduction"]
    assert len(intro) >= 2
    assert all(len(c.text.split()) <= 40 for c in intro)
    assert intro[0].text.split()[-10:] == intro[1].text.split()[:10]


def test_chunk_ids_unique_and_stable(chunks, tmp_path):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    pdf = tmp_path / "again.pdf"
    make_paper(pdf)
    assert [c.chunk_id for c in ingest_pdf(pdf, "toy", size=40, overlap=10)] == ids


def test_normalize_handles_ligatures_and_quotes():
    assert normalize("eﬃcient “DTR”  re-\nmaterialize") == 'effiient "DTR" rematerialize'.replace(
        "effiient", "eﬃcient"
    )


def test_section_words_joins_line_end_hyphen():
    from papertrail.ingest import Line, Section

    sec = Section("x", [Line("lowest heuris-", 2, 10, False), Line("tic score", 3, 10, False)])
    assert section_words(sec) == [("lowest", 2), ("heuristic", 2), ("score", 3)]


def test_evidence_survives_rechunking(tmp_path):
    pdf = tmp_path / "toy.pdf"
    make_paper(pdf)
    quote = "picks the tensor with the lowest heuristic score and frees it immediately"
    q = GoldQuestion(
        id="q1",
        question="Which tensor is evicted?",
        kind="factual",
        evidence=[Evidence(paper_id="toy", page=2, quote=quote)],
    )
    for size, overlap in [(40, 10), (15, 3), (300, 50)]:
        index = EvidenceIndex(ingest_pdf(pdf, "toy", size=size, overlap=overlap))
        assert index.gold_chunk_ids(q), f"quote lost at chunk size {size}"
        assert not index.unresolved([q])


def test_unresolved_reports_typos_and_unknown_papers(chunks):
    index = EvidenceIndex(chunks)
    bad = GoldQuestion(
        id="q2",
        question="?",
        kind="factual",
        evidence=[
            Evidence(
                paper_id="toy", page=1, quote="this sentence is nowhere in the toy paper at all"
            ),
            Evidence(paper_id="nope", page=1, quote="some quote that is long enough"),
        ],
    )
    problems = index.unresolved([bad])
    assert len(problems) == 2 and "not in the corpus" in problems[1]


def test_squash_ignores_hyphens_and_spacing():
    assert squash("memory-\nefficient  Training") == squash("memoryefficient training")
