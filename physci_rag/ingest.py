import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .benchmark import file_to_record_ids, load_benchmark
from .config import CHUNK_OVERLAP, CHUNK_SIZE, FILES_DIR


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source_file: str
    page: int | None
    record_ids: list[str]
    chunk_index: int


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_pdf_text(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _normalize_whitespace(page.extract_text() or "")
        if text:
            pages.append((page_number, text))
    return pages


def chunk_pdf(
    path: Path,
    record_ids: list[str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    source_file = path.name
    chunks: list[DocumentChunk] = []
    chunk_counter = 0

    for page_number, page_text in extract_pdf_text(path):
        for piece in _split_text(page_text, chunk_size, overlap):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{source_file}::p{page_number}::c{chunk_counter}",
                    text=piece,
                    source_file=source_file,
                    page=page_number,
                    record_ids=record_ids,
                    chunk_index=chunk_counter,
                )
            )
            chunk_counter += 1

    return chunks


def ingest_local_files(files_dir: Path = FILES_DIR) -> list[DocumentChunk]:
    records = load_benchmark()
    mapping = file_to_record_ids(records)
    all_chunks: list[DocumentChunk] = []

    pdf_paths = sorted(files_dir.glob("*.pdf"))
    for path in pdf_paths:
        record_ids = mapping.get(path.name, [])
        all_chunks.extend(chunk_pdf(path, record_ids))

    return all_chunks
