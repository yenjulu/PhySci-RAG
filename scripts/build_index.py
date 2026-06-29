import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from physci_rag.download import download_files
from physci_rag.ingest import ingest_local_files
from physci_rag.store import VectorStore


def main() -> None:
  parser = argparse.ArgumentParser(description="Build the PhySciBench RAG index.")
  parser.add_argument(
    "--download-limit",
    type=int,
    default=None,
    help="Download only the first N referenced source files (useful for quick testing).",
  )
  parser.add_argument(
    "--pdf-only",
    action="store_true",
    help="Download and index PDF text only, skip image figures.",
  )
  parser.add_argument(
    "--skip-download",
    action="store_true",
    help="Only ingest source files already present in data/files/.",
  )
  parser.add_argument(
    "--no-ocr",
    action="store_true",
    help="Skip OCR when indexing image figures.",
  )
  args = parser.parse_args()

  if not args.skip_download:
    download_files(limit=args.download_limit, pdf_only=args.pdf_only)

  chunks = ingest_local_files(include_images=not args.pdf_only, use_ocr=not args.no_ocr)
  text_count = sum(1 for chunk in chunks if chunk.content_type == "text")
  image_count = sum(1 for chunk in chunks if chunk.content_type == "image")
  print(
    f"Ingested {len(chunks)} chunks "
    f"({text_count} text, {image_count} image) from local source files."
  )

  store = VectorStore()
  store.build(chunks)
  store.save()
  print("Index saved to data/index/")


if __name__ == "__main__":
  main()
