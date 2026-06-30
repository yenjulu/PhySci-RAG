import re
from dataclasses import dataclass
from pathlib import Path

import fitz

from .benchmark import file_to_record_ids, load_benchmark
from .config import CHUNK_OVERLAP, CHUNK_SIZE, FILES_DIR, IMAGE_EXTENSIONS, OCR_ENABLED


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source_file: str
    page: int | None
    record_ids: list[str]
    chunk_index: int
    content_type: str = "text"


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_ACRONYM_GLUE = re.compile(r"([a-z]{4,})([A-Z][A-Z0-9]+)")
_STUCK_LABEL = re.compile(r"(\w)(Fig\.|Table\.|Eq\.)")
_BRACKET_GLUE = re.compile(r"([\)\]\}])([A-Za-z])")
_HYPHEN_BREAK = re.compile(r"(\w)-\s+(\w)")


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _repair_pdf_text(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _BRACKET_GLUE.sub(r"\1 \2", text)
    text = _ACRONYM_GLUE.sub(r"\1 \2", text)
    text = _STUCK_LABEL.sub(r"\1 \2", text)
    return _normalize_whitespace(text)


def _split_oversized_segment(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    pieces: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:
        add = len(word) + (1 if current else 0)
        if current and length + add > chunk_size:
            pieces.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += add

    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_sentences(text: str, chunk_size: int) -> list[str]:
    sentences: list[str] = []
    for part in _SENTENCE_BOUNDARY.split(text):
        part = part.strip()
        if not part:
            continue
        if len(part) <= chunk_size:
            sentences.append(part)
        else:
            sentences.extend(_split_oversized_segment(part, chunk_size))
    return sentences


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    units = _split_sentences(text, chunk_size)
    if not units:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(units):
        length = 0
        end = start
        while end < len(units):
            piece = units[end]
            add = len(piece) + (1 if end > start else 0)
            if length + add > chunk_size and end > start:
                break
            length += add
            end += 1

        chunk = " ".join(units[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(units):
            break

        if overlap <= 0:
            start = end
            continue

        overlap_start = end
        overlap_len = 0
        while overlap_start > start:
            overlap_start -= 1
            piece = units[overlap_start]
            overlap_len += len(piece) + (1 if overlap_start < end - 1 else 0)
            if overlap_len >= overlap:
                break

        next_start = max(overlap_start, start + 1)
        if next_start >= end:
            start = end
        else:
            start = next_start

    return chunks


def extract_pdf_text(path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with fitz.open(str(path)) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = _repair_pdf_text(page.get_text("text") or "")
            if text:
                pages.append((page_number, text))
    return pages


def chunk_pdf(
    path: Path,
    record_ids: list[str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    chunk_index_start: int = 0,
) -> list[DocumentChunk]:
    source_file = path.name
    chunks: list[DocumentChunk] = []
    chunk_counter = chunk_index_start

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
                    content_type="text",
                )
            )
            chunk_counter += 1

    return chunks


def _image_search_text(source_file: str, record_ids: list[str], ocr_text: str = "") -> str:
    ids = ", ".join(record_ids) if record_ids else "unlinked"
    header = f"[image] {source_file} | benchmark records {ids}"
    if ocr_text:
        return f"{header}\n{ocr_text}"
    return (
        f"{header}\n"
        "scientific figure microscopy diffraction spectroscopy structure"
    )


def chunk_image(
    path: Path,
    record_ids: list[str],
    chunk_index: int,
    use_ocr: bool = OCR_ENABLED,
) -> DocumentChunk:
    source_file = path.name
    ocr_text = ""
    if use_ocr:
        from .ocr import extract_image_text

        ocr_text = extract_image_text(path)

    return DocumentChunk(
        chunk_id=f"{source_file}::img::c{chunk_index}",
        text=_image_search_text(source_file, record_ids, ocr_text),
        source_file=source_file,
        page=None,
        record_ids=record_ids,
        chunk_index=chunk_index,
        content_type="image",
    )


def ingest_local_files(
    files_dir: Path = FILES_DIR,
    include_images: bool = True,
    use_ocr: bool = OCR_ENABLED,
) -> list[DocumentChunk]:
    records = load_benchmark()
    mapping = file_to_record_ids(records)
    all_chunks: list[DocumentChunk] = []
    chunk_counter = 0

    pdf_paths = sorted(files_dir.glob("*.pdf"))
    for path in pdf_paths:
        record_ids = mapping.get(path.name, [])
        pdf_chunks = chunk_pdf(path, record_ids, chunk_index_start=chunk_counter)
        all_chunks.extend(pdf_chunks)
        chunk_counter += len(pdf_chunks)

    if include_images:
        image_paths = sorted(
            path
            for path in files_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for path in image_paths:
            record_ids = mapping.get(path.name, [])
            all_chunks.append(chunk_image(path, record_ids, chunk_counter, use_ocr=use_ocr))
            chunk_counter += 1

    return all_chunks
