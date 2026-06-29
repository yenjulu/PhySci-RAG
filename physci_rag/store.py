from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import (
  EMBEDDING_MODEL,
  FILES_DIR,
  HYBRID_ALPHA,
  IMAGE_EMBEDDING_MODEL,
  INDEX_DIR,
)
from .ingest import DocumentChunk


def _chunk_from_dict(item: dict) -> DocumentChunk:
  return DocumentChunk(
    chunk_id=item["chunk_id"],
    text=item["text"],
    source_file=item["source_file"],
    page=item["page"],
    record_ids=item["record_ids"],
    chunk_index=item["chunk_index"],
    content_type=item.get("content_type", "text"),
  )


class VectorStore:
  def __init__(self, index_dir: Path = INDEX_DIR):
    self.index_dir = index_dir
    self.chunks: list[DocumentChunk] = []
    self.text_embeddings: np.ndarray | None = None
    self.image_embeddings: np.ndarray | None = None
    self.bm25: BM25Okapi | None = None
    self.text_model: SentenceTransformer | None = None
    self.image_model: SentenceTransformer | None = None

  def _embedding_dim(self, model: SentenceTransformer, sample: str | Image.Image) -> int:
    dim = model.get_sentence_embedding_dimension()
    if dim is not None:
      return dim
    encoded = model.encode(sample, normalize_embeddings=True)
    return int(encoded.shape[-1])

  def _tokenize(self, text: str) -> list[str]:
    return text.lower().split()

  def build(
    self,
    chunks: list[DocumentChunk],
    model_name: str = EMBEDDING_MODEL,
    image_model_name: str = IMAGE_EMBEDDING_MODEL,
    files_dir: Path = FILES_DIR,
  ) -> None:
    if not chunks:
      raise ValueError("No chunks to index. Download source files first.")

    self.chunks = chunks
    self.text_model = SentenceTransformer(model_name)
    self.image_model = SentenceTransformer(image_model_name)

    text_indices = [idx for idx, chunk in enumerate(chunks) if chunk.content_type == "text"]
    image_indices = [idx for idx, chunk in enumerate(chunks) if chunk.content_type == "image"]

    text_dim = self._embedding_dim(self.text_model, "dimension probe")
    image_dim = self._embedding_dim(
      self.image_model,
      Image.new("RGB", (32, 32), color="white"),
    )
    text_embeddings = np.zeros((len(chunks), text_dim), dtype=np.float32)
    image_embeddings = np.zeros((len(chunks), image_dim), dtype=np.float32)

    if text_indices:
      texts = [chunks[idx].text for idx in text_indices]
      encoded = self.text_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
      )
      for row, chunk_idx in enumerate(text_indices):
        text_embeddings[chunk_idx] = encoded[row]

    if image_indices:
      images: list[Image.Image] = []
      for chunk_idx in image_indices:
        path = files_dir / chunks[chunk_idx].source_file
        if not path.exists():
          raise FileNotFoundError(f"Image not found for indexing: {path}")
        images.append(Image.open(path).convert("RGB"))

      encoded = self.image_model.encode(
        images,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True,
      )
      for row, chunk_idx in enumerate(image_indices):
        image_embeddings[chunk_idx] = encoded[row]

    self.text_embeddings = text_embeddings
    self.image_embeddings = image_embeddings
    tokenized = [self._tokenize(chunk.text) for chunk in chunks]
    self.bm25 = BM25Okapi(tokenized)

  def save(self) -> None:
    if self.text_embeddings is None or self.image_embeddings is None or self.bm25 is None:
      raise RuntimeError("Index is empty. Run build() first.")

    self.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(self.index_dir / "embeddings.npy", self.text_embeddings)
    np.save(self.index_dir / "image_embeddings.npy", self.image_embeddings)
    metadata = [asdict(chunk) for chunk in self.chunks]
    (self.index_dir / "chunks.json").write_text(
      json.dumps(metadata, ensure_ascii=False, indent=2),
      encoding="utf-8",
    )
    (self.index_dir / "meta.json").write_text(
      json.dumps(
        {
          "embedding_model": EMBEDDING_MODEL,
          "image_embedding_model": IMAGE_EMBEDDING_MODEL,
        },
        indent=2,
      ),
      encoding="utf-8",
    )

  def load(
    self,
    model_name: str = EMBEDDING_MODEL,
    image_model_name: str = IMAGE_EMBEDDING_MODEL,
  ) -> None:
    embeddings_path = self.index_dir / "embeddings.npy"
    image_embeddings_path = self.index_dir / "image_embeddings.npy"
    chunks_path = self.index_dir / "chunks.json"
    if not embeddings_path.exists() or not chunks_path.exists():
      raise FileNotFoundError(
        f"Index not found in {self.index_dir}. Run `py -3 scripts/build_index.py` first."
      )

    self.text_embeddings = np.load(embeddings_path)
    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    self.chunks = [_chunk_from_dict(item) for item in raw_chunks]

    if image_embeddings_path.exists():
      self.image_embeddings = np.load(image_embeddings_path)
    else:
      probe_model = SentenceTransformer(image_model_name)
      image_dim = probe_model.get_sentence_embedding_dimension()
      if image_dim is None:
        image_dim = int(
          probe_model.encode(
            Image.new("RGB", (32, 32), color="white"),
            normalize_embeddings=True,
          ).shape[-1]
        )
      self.image_embeddings = np.zeros((len(self.chunks), image_dim), dtype=np.float32)

    tokenized = [self._tokenize(chunk.text) for chunk in self.chunks]
    self.bm25 = BM25Okapi(tokenized)
    self.text_model = SentenceTransformer(model_name)
    self.image_model = SentenceTransformer(image_model_name)

  @property
  def ready(self) -> bool:
    return (
      self.text_embeddings is not None
      and self.image_embeddings is not None
      and self.bm25 is not None
      and self.text_model is not None
      and self.image_model is not None
    )

  def _dense_scores(self, query: str) -> np.ndarray:
    assert (
      self.text_model is not None
      and self.image_model is not None
      and self.text_embeddings is not None
      and self.image_embeddings is not None
    )

    text_query = self.text_model.encode(query, normalize_embeddings=True)
    image_query = self.image_model.encode(query, normalize_embeddings=True)
    text_scores = self.text_embeddings @ text_query
    image_scores = self.image_embeddings @ image_query
    is_image = np.array([chunk.content_type == "image" for chunk in self.chunks])
    return np.where(is_image, image_scores, text_scores)

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
