from __future__ import annotations

from dataclasses import dataclass

from .benchmark import BenchmarkRecord, load_benchmark
from .config import TOP_K
from .generator import RetrievedContext, format_context, generate_answer
from .ingest import DocumentChunk
from .store import VectorStore


@dataclass(frozen=True)
class RAGResult:
  question: str
  answer: str
  contexts: list[RetrievedContext]
  benchmark_id: str | None = None
  ground_truth: str | None = None
  task_type: str | None = None


class PhySciRAG:
  def __init__(self, store: VectorStore | None = None):
    self.store = store or VectorStore()
    self.benchmark = load_benchmark()

  def ensure_index(self) -> None:
    if not self.store.ready:
      self.store.load()

  def get_record(self, record_id: str) -> BenchmarkRecord:
    for record in self.benchmark:
      if record.id == record_id:
        return record
    raise KeyError(f"Unknown benchmark id: {record_id}")

  def query(
    self,
    question: str,
    top_k: int = TOP_K,
    record_id: str | None = None,
    use_llm: bool = True,
  ) -> RAGResult:
    self.ensure_index()

    source_files: set[str] | None = None
    ground_truth = None
    task_type = None
    if record_id:
      record = self.get_record(record_id)
      question = record.question
      ground_truth = record.answer
      task_type = record.type
      if record.files:
        source_files = set(record.files)

    hits = self.store.search(question, top_k=top_k, source_files=source_files)
    contexts = [RetrievedContext(chunk=chunk, score=score) for chunk, score in hits]
    answer = generate_answer(question, contexts) if use_llm else format_context(contexts)

    return RAGResult(
      question=question,
      answer=answer,
      contexts=contexts,
      benchmark_id=record_id,
      ground_truth=ground_truth,
      task_type=task_type,
    )

  def list_questions(self, task_type: str | None = None, limit: int = 10) -> list[BenchmarkRecord]:
    records = self.benchmark
    if task_type:
      records = [record for record in records if record.type == task_type]
    return records[:limit]
