from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, HYBRID_ALPHA, INDEX_DIR
from .ingest import DocumentChunk


class VectorStore:
  def __init__(self, index_dir: Path = INDEX_DIR):
    self.index_dir = index_dir
    self.chunks: list[DocumentChunk] = []
    self.embeddings: np.ndarray | None = None
    self.bm25: BM25Okapi | None = None
    self.model: SentenceTransformer | None = None

  def _tokenize(self, text: str) -> list[str]:
    return text.lower().split()

  def build(self, chunks: list[DocumentChunk], model_name: str = EMBEDDING_MODEL) -> None:
    if not chunks:
      raise ValueError("No chunks to index. Download PDFs first.")

    self.model = SentenceTransformer(model_name)
    texts = [chunk.text for chunk in chunks]
    self.chunks = chunks
    self.embeddings = self.model.encode(
      texts,
      batch_size=32,
      show_progress_bar=True,
      normalize_embeddings=True,
    )
    tokenized = [self._tokenize(text) for text in texts]
    self.bm25 = BM25Okapi(tokenized)

  def save(self) -> None:
    if self.embeddings is None or self.bm25 is None:
      raise RuntimeError("Index is empty. Run build() first.")

    self.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(self.index_dir / "embeddings.npy", self.embeddings)
    metadata = [asdict(chunk) for chunk in self.chunks]
    (self.index_dir / "chunks.json").write_text(
      json.dumps(metadata, ensure_ascii=False, indent=2),
      encoding="utf-8",
    )
    (self.index_dir / "meta.json").write_text(
      json.dumps({"embedding_model": EMBEDDING_MODEL}, indent=2),
      encoding="utf-8",
    )

  def load(self, model_name: str = EMBEDDING_MODEL) -> None:
    embeddings_path = self.index_dir / "embeddings.npy"
    chunks_path = self.index_dir / "chunks.json"
    if not embeddings_path.exists() or not chunks_path.exists():
      raise FileNotFoundError(
        f"Index not found in {self.index_dir}. Run `py -3 scripts/build_index.py` first."
      )

    self.embeddings = np.load(embeddings_path)
    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    self.chunks = [DocumentChunk(**item) for item in raw_chunks]
    tokenized = [self._tokenize(chunk.text) for chunk in self.chunks]
    self.bm25 = BM25Okapi(tokenized)
    self.model = SentenceTransformer(model_name)

  @property
  def ready(self) -> bool:
    return self.embeddings is not None and self.bm25 is not None and self.model is not None

  def _dense_scores(self, query: str) -> np.ndarray:
    assert self.model is not None and self.embeddings is not None
    query_vec = self.model.encode(query, normalize_embeddings=True)
    return self.embeddings @ query_vec

  def _sparse_scores(self, query: str) -> np.ndarray:
    assert self.bm25 is not None
    tokens = self._tokenize(query)
    scores = np.array(self.bm25.get_scores(tokens), dtype=np.float32)
    if scores.max() > 0:
      scores = scores / scores.max()
    return scores

  def search(
    self,
    query: str,
    top_k: int = 6,
    source_files: set[str] | None = None,
    alpha: float = HYBRID_ALPHA,
  ) -> list[tuple[DocumentChunk, float]]:
    if not self.ready:
      raise RuntimeError("Vector store is not loaded.")

    dense = self._dense_scores(query)
    sparse = self._sparse_scores(query)
    combined = alpha * dense + (1.0 - alpha) * sparse

    if source_files:
      mask = np.array([chunk.source_file in source_files for chunk in self.chunks])
      combined = np.where(mask, combined, combined.min() - 1.0)

    ranked_idx = np.argsort(combined)[::-1][:top_k]
    return [(self.chunks[i], float(combined[i])) for i in ranked_idx]
