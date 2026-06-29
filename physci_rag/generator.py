from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

from .config import LLM_MODEL, MAX_CONTEXT_CHARS
from .ingest import DocumentChunk


@dataclass(frozen=True)
class RetrievedContext:
  chunk: DocumentChunk
  score: float


SYSTEM_PROMPT = """You are a physical-sciences research assistant.
Answer using only the provided context from scientific papers.
If the context is insufficient, say what is missing instead of guessing.
Be precise with terminology, units, and experimental details."""


def format_context(contexts: list[RetrievedContext]) -> str:
  blocks: list[str] = []
  used = 0
  for idx, item in enumerate(contexts, start=1):
    header = f"[{idx}] {item.chunk.source_file}"
    if item.chunk.content_type == "image":
      header += " (image figure)"
    elif item.chunk.page is not None:
      header += f" (page {item.chunk.page})"
    block = f"{header}\n{item.chunk.text}"
    if used + len(block) > MAX_CONTEXT_CHARS:
      break
    blocks.append(block)
    used += len(block)
  return "\n\n".join(blocks)


def generate_answer(
  question: str,
  contexts: list[RetrievedContext],
  model: str = LLM_MODEL,
) -> str:
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    return _fallback_answer(question, contexts)

  client = OpenAI(api_key=api_key)
  context_text = format_context(contexts)
  response = client.chat.completions.create(
    model=model,
    temperature=0.1,
    messages=[
      {"role": "system", "content": SYSTEM_PROMPT},
      {
        "role": "user",
        "content": (
          f"Question:\n{question}\n\n"
          f"Context:\n{context_text}\n\n"
          "Provide a concise, evidence-grounded answer."
        ),
      },
    ],
  )
  return response.choices[0].message.content or ""


def _fallback_answer(question: str, contexts: list[RetrievedContext]) -> str:
  if not contexts:
    return "No relevant context retrieved. Set OPENAI_API_KEY for generation, or rebuild the index."

  lines = [
    "OPENAI_API_KEY is not set, so this is a retrieval-only response.",
    f"Question: {question}",
    "",
    "Top retrieved passages:",
  ]
  for idx, item in enumerate(contexts, start=1):
    preview = item.chunk.text[:500]
    if len(item.chunk.text) > 500:
      preview += "..."
    location = item.chunk.source_file
    if item.chunk.content_type == "image":
      location += ", image figure"
    elif item.chunk.page is not None:
      location += f", page {item.chunk.page}"
    lines.append(f"{idx}. [{location}, score={item.score:.3f}] {preview}")
  return "\n".join(lines)
