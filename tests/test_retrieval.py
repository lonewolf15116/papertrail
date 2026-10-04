import os
from pathlib import Path

import pytest

from papertrail.embed import HashEmbedder, OverlapReranker
from papertrail.eval.evidence import EvidenceIndex
from papertrail.eval.retrieval_eval import evaluate, markdown_table, scored_questions
from papertrail.retrieval import (
    BM25Retriever,
    HybridRetriever,
    RerankRetriever,
    reciprocal_rank_fusion,
    tokenize,
)
from papertrail.schemas import Chunk, Evidence, GoldQuestion

REPO = Path(__file__).resolve().parents[1]

CHUNKS = [
    Chunk(
        chunk_id="dtr:0",
        paper_id="dtr",
        section="2",
        page=3,
        text="DTR evicts the tensor that is stalest, largest and cheapest to recompute",
    ),
    Chunk(
        chunk_id="dtr:1",
        paper_id="dtr",
        section="2",
        page=3,
        text="Banishing permanently frees deallocated tensors and constants",
    ),
    Chunk(
        chunk_id="ckm:0",
        paper_id="checkmate",
        section="4",
        page=6,
        text="Checkmate solves an integer linear program offline with an MILP solver",
    ),
    Chunk(
        chunk_id="ckm:1",
        paper_id="checkmate",
        section="1",
        page=1,
        text="Rematerialization frees tensors and recomputes them, trading compute for memory",
    ),
    Chunk(
        chunk_id="zero:0",
        paper_id="zero",
        section="5",
        page=10,
        text="ZeRO partitions optimizer states across data parallel processes",
    ),
]


class ListRetriever:
    def __init__(self, name, ids):
        self.name = name
        self._ids = ids
        self._by = {c.chunk_id: c for c in CHUNKS}

    def retrieve(self, question, k):
        return [self._by[i] for i in self._ids[:k]]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("Which tensor does DTR's heuristic evict?") == [
        "tensor",
        "dtr",
        "s",
        "heuristic",
        "evict",
    ]


def test_bm25_ranks_the_matching_chunk_first():
    r = BM25Retriever(CHUNKS)
    assert r.retrieve("which solver does Checkmate use offline", 2)[0].chunk_id == "ckm:0"
    assert r.retrieve("optimizer state partitioning", 1)[0].chunk_id == "zero:0"


def test_hybrid_fuses_and_dedupes():
    a = ListRetriever("a", ["dtr:0", "ckm:0", "zero:0"])
    b = ListRetriever("b", ["ckm:0", "dtr:1", "dtr:0"])
    got = [c.chunk_id for c in HybridRetriever([a, b], candidates=3).retrieve("q", 4)]
    assert got[:2] == ["ckm:0", "dtr:0"]  # in both lists
    assert len(got) == len(set(got)) == 4


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert fused[:2] == ["a", "b"] and set(fused) == {"a", "b", "c", "d"}


def test_rerank_reorders_candidates():
    base = ListRetriever("base", ["zero:0", "dtr:1", "ckm:0"])
    r = RerankRetriever(base, OverlapReranker(), candidates=3)
    assert r.retrieve("checkmate milp solver offline", 1)[0].chunk_id == "ckm:0"
    assert r.name == "base_rerank"


def test_hash_embedder_is_normalised_and_stable():
    e = HashEmbedder(64)
    a, b = e.embed_queries(["tensor eviction", "tensor eviction"])
    assert abs(float((a * a).sum()) - 1.0) < 1e-5 and (a == b).all()


def _q(qid, kind, ev, distract=()):
    return GoldQuestion(
        id=qid,
        question=qid,
        kind=kind,
        status="draft",
        evidence=[Evidence(paper_id=p, page=1, quote=t) for p, t in ev],
        distractor_paper_ids=list(distract),
    )


def test_evaluate_scores_recall_mrr_and_wrong_paper():
    index = EvidenceIndex(CHUNKS)
    questions = [
        _q("f1", "factual", [("dtr", "evicts the tensor that is stalest, largest and cheapest")]),
        _q(
            "x1",
            "cross_paper",
            [
                ("dtr", "evicts the tensor that is stalest, largest and cheapest"),
                ("checkmate", "solves an integer linear program offline"),
            ],
            distract=["zero"],
        ),
    ]
    r = ListRetriever("fixed", ["zero:0", "dtr:0", "dtr:1", "ckm:1", "ckm:0"])
    res = evaluate(r, questions, index, {c.chunk_id: c.paper_id for c in CHUNKS}.__getitem__, True)
    by = {q.id: q for q in res.questions}
    assert by["f1"].recall_at_5 == 1.0 and by["f1"].rank == 2
    assert by["x1"].recall_at_5 == 1.0 and by["x1"].wrong_paper_at_5 == pytest.approx(0.2)
    assert res.mrr_at_10 == pytest.approx(0.5)
    assert res.provisional and "| fixed |" in markdown_table([res])


def test_scored_questions_excludes_drafts_and_unanswerable():
    v = GoldQuestion(
        id="v",
        question="?",
        kind="factual",
        evidence=[Evidence(paper_id="dtr", page=1, quote="a long enough quote here")],
    )
    d = v.model_copy(update={"id": "d", "status": "draft"})
    u = GoldQuestion(id="u", question="?", kind="unanswerable")
    assert [q.id for q in scored_questions([v, d, u], include_drafts=False)] == ["v"]
    assert [q.id for q in scored_questions([v, d, u], include_drafts=True)] == ["v", "d"]


# --- Postgres/pgvector: runs when a test database is configured (CI provides one) ----------

DB_URL = os.environ.get("PAPERTRAIL_TEST_DATABASE_URL")
needs_db = pytest.mark.skipif(not DB_URL, reason="PAPERTRAIL_TEST_DATABASE_URL not set")


@pytest.fixture
def conn():
    from papertrail.index import connect

    c = connect(DB_URL)
    with c.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS papers;")
        cur.execute((REPO / "scripts" / "init.sql").read_text())
    yield c
    c.close()


PAPERS = [{"id": p, "title": p.upper()} for p in ("dtr", "checkmate", "zero")]


@needs_db
def test_index_is_idempotent_and_removes_stale_chunks(conn):
    from papertrail.index import index_chunks

    stats = index_chunks(conn, CHUNKS, PAPERS, HashEmbedder(384))
    assert stats == {"chunks": 5, "embedded": 5, "removed": 0}
    stats = index_chunks(conn, CHUNKS[:3], PAPERS, HashEmbedder(384))
    assert stats["chunks"] == 3 and stats["removed"] == 2


@needs_db
def test_index_rejects_wrong_embedding_width(conn):
    from papertrail.index import index_chunks

    with pytest.raises(ValueError, match="VECTOR"):
        index_chunks(conn, CHUNKS, PAPERS, HashEmbedder(128))


@needs_db
def test_pg_fulltext_and_vector_retrievers(conn):
    from papertrail.index import index_chunks
    from papertrail.retrieval import PgFullTextRetriever, PgVectorRetriever

    index_chunks(conn, CHUNKS, PAPERS, HashEmbedder(384))
    fts = PgFullTextRetriever(conn).retrieve("Which MILP solver does Checkmate use?", 3)
    assert fts[0].chunk_id == "ckm:0"
    assert PgFullTextRetriever(conn).retrieve("the of and", 3) == []  # only stopwords
    vec = PgVectorRetriever(conn, HashEmbedder(384)).retrieve("partitions optimizer states", 2)
    assert vec[0].chunk_id == "zero:0"


def test_bm25_header_adds_paper_identity():
    titles = {
        "dtr": "Dynamic Tensor Rematerialization",
        "checkmate": "Checkmate",
        "zero": "ZeRO Memory Optimizations",
    }
    plain = BM25Retriever(CHUNKS)
    headed = BM25Retriever(CHUNKS, titles=titles)
    assert headed.name == "bm25_header"
    assert headed.retrieve("memory optimizations", 1)[0].chunk_id == "zero:0"
    assert plain.retrieve("memory optimizations", 1)[0].chunk_id != "zero:0"
