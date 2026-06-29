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
    help="Download only the first N referenced PDFs (useful for quick testing).",
  )
  parser.add_argument(
    "--include-images",
    action="store_true",
    help="Also download image files (not used for indexing).",
  )
  parser.add_argument(
    "--skip-download",
    action="store_true",
    help="Only ingest PDFs already present in data/files/.",
  )
  args = parser.parse_args()

  if not args.skip_download:
    download_files(limit=args.download_limit, pdf_only=not args.include_images)

  chunks = ingest_local_files()
  print(f"Ingested {len(chunks)} chunks from local PDFs.")

  store = VectorStore()
  store.build(chunks)
  store.save()
  print("Index saved to data/index/")


if __name__ == "__main__":
  main()
