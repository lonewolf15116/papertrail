"""Embedding and reranking models behind small protocols, so tests use fakes and the
retrievers never import torch directly.

Models load from a local folder (models/<org>__<name>, filled by scripts/download_models.py)
when it exists, otherwise from the Hugging Face id.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]

# bge v1.5 recommends this prefix on short queries (not passages) for retrieval.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int

    def embed_queries(self, texts: Sequence[str]) -> Vectors: ...

    def embed_passages(self, texts: Sequence[str]) -> Vectors: ...


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


def resolve_model(name: str, models_dir: Path = Path("models")) -> str:
    local = models_dir / name.replace("/", "__")
    return str(local) if local.exists() else name


class SentenceTransformerEmbedder:
    def __init__(self, model: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model
        self._model = SentenceTransformer(resolve_model(model), device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)
        self._batch = batch_size
        self._prefix = BGE_QUERY_PREFIX if "bge" in model.lower() else ""

    def _encode(self, texts: Sequence[str]) -> Vectors:
        out = self._model.encode(
            list(texts), batch_size=self._batch, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(out, dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        return self._encode([self._prefix + t for t in texts])

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        return self._encode(texts)


class CrossEncoderReranker:
    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model
        self._model = CrossEncoder(resolve_model(model), device="cpu")

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        scores = self._model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]


class HashEmbedder:
    """Deterministic bag-of-words embedder for tests: no model download, stable vectors.

    Texts sharing words get similar vectors, which is enough to test the plumbing."""

    def __init__(self, dim: int = 384) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim

    def _one(self, text: str) -> Vectors:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._one(t) for t in texts]).astype(np.float32)

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        return self.embed_queries(texts)


class OverlapReranker:
    """Test reranker: scores by shared words with the query."""

    name = "overlap"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        return [float(len(q & set(re.findall(r"[a-z0-9]+", p.lower())))) for p in passages]
