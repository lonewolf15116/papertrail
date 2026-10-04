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


def test_squash_expands_ligatures():  # PDF text and viewer copies use U+FB01 / U+FB02
    assert squash("pro\ufb01ling the o\ufb04ine bound") == squash("profiling the offline bound")


# --- Heading layouts seen in the real corpus -------------------------------------------------

from papertrail.ingest import Line, split_sections  # noqa: E402


def _titles(rows: list[tuple[str, float, bool]], page: int = 1) -> list[str]:
    body = [("Body text " * 8, 10.0, False)] * 3
    lines = []
    for text, size, bold in rows:
        lines.append(Line(text, page, size, bold))
        lines.extend(Line(t, page, s, b) for t, s, b in body)
    return [s.title for s in split_sections(lines)]


def test_number_and_title_on_separate_lines_are_joined():  # ICLR / MLSys (DTR, Checkmate)
    body = [Line("Body text " * 8, 1, 10.0, False)] * 3
    lines = [Line("1", 1, 12.0, False), Line("INTRODUCTION", 1, 12.0, False), *body]
    lines += [Line("2.1", 1, 10.0, True), Line("Problem definition", 1, 10.0, True), *body]
    titles = [s.title for s in split_sections(lines)]
    assert titles == ["1 Introduction", "2.1 Problem definition"]


def test_ieee_roman_headings_without_bold():  # vDNN
    titles = _titles(
        [("I. INTRODUCTION", 10.0, False), ("II. BACKGROUND AND MOTIVATION", 10.0, False)]
    )
    assert titles == ["I. Introduction", "II. Background and motivation"]


def test_all_caps_subsections_without_bold():  # ICLR appendix "A.1 NETWORK DEFINITION"
    titles = _titles([("1 INTRODUCTION", 12.0, False), ("A.1 NETWORK DEFINITION", 10.0, False)])
    assert titles == ["1 Introduction", "A.1 Network definition"]


def test_numbered_affiliations_are_front_matter():  # Chen et al. 2016
    titles = _titles(
        [
            ("1 University of Washington", 10.0, True),
            ("3 Massachusetts Institute of Technology", 10.0, True),
            ("1 Introduction", 12.0, True),
        ]
    )
    assert titles == ["Front matter", "1 Introduction"]


def test_pseudocode_and_table_numbers_rejected():  # Gruslys pseudocode, 8-bit "9.0"
    titles = _titles(
        [
            ("1 Introduction", 12.0, True),
            ("2 Method", 12.0, True),
            ("3 Analysis", 12.0, True),
            ("1 Order of execution", 10.0, True),
            ("9.0 Ablation Analysis", 10.0, True),
            ("4 Results", 12.0, True),
        ]
    )
    assert titles == ["1 Introduction", "2 Method", "3 Analysis", "4 Results"]


def test_small_bold_figure_labels_are_not_headings():
    titles = _titles([("1 Introduction", 12.0, True), ("2 Memory Ratio", 6.0, True)])
    assert titles == ["1 Introduction"]


def test_quote_pages_and_fix_pages(tmp_path):
    from papertrail.eval.evidence import fix_pages, quote_pages

    pdf = tmp_path / "toy.pdf"
    make_paper(pdf)
    assert quote_pages(pdf, "Appendix text about optimal schedules stays in the corpus") == [4]
    assert quote_pages(pdf, "this sentence does not appear anywhere in it") == []
    q = GoldQuestion(
        id="q",
        question="?",
        kind="factual",
        evidence=[Evidence(paper_id="toy", page=1, quote="Appendix text about optimal schedules")],
    )
    changes = fix_pages([q], tmp_path)
    assert q.evidence[0].page == 4 and changes == ["q: toy page 1 -> 4"]
