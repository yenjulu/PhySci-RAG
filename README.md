# PhySci-RAG

A moderate retrieval-augmented generation (RAG) pipeline built on the [PhySciBench](https://huggingface.co/datasets/yigengx/PhySciBench) dataset for physical-sciences deep research.

## What it does

1. Loads the 200 PhySciBench benchmark records from `data/physcibench.json`
2. Downloads referenced source PDFs from Hugging Face
3. Chunks and indexes paper text plus benchmark figure images with hybrid retrieval (dense embeddings + BM25)
4. Retrieves relevant passages for a question
5. Generates an answer with an OpenAI-compatible LLM (or returns retrieval-only output without an API key)

## Architecture

```
physcibench.json + files/*
        │
        ▼
   PDF chunking (pypdf) + image indexing (CLIP)
        │
        ▼
  Embeddings (MiniLM for text, CLIP for images) + BM25 index
        │
        ▼
  Hybrid retrieval (top-k)
        │
        ▼
  LLM answer generation (optional)
```

## Setup

```powershell
cd PhySci-RAG
py -3 -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you want generated answers. Without it, `query.py` still returns retrieved passages.

The benchmark JSON is already included in `data/physcibench.json`.

## Build the index

Download source PDFs/images and build the vector index:

```powershell
# Quick test with the first 10 source files (~few minutes)
py -3 scripts/build_index.py --download-limit 10

# Full index (123 PDFs + 18 images — takes longer)
py -3 scripts/build_index.py

# PDF text only (skip image figures)
py -3 scripts/build_index.py --pdf-only
```

Re-run ingestion only (if PDFs are already downloaded):

```powershell
py -3 scripts/build_index.py --skip-download
```

## Query

Ask a free-form question:

```powershell
py -3 scripts/query.py "What dopants were studied in CsV3Sb5 STM experiments?"
```

Run against a specific benchmark item (uses its referenced files as a retrieval filter):

```powershell
py -3 scripts/query.py --id physci-001
```

Retrieval only (no LLM):

```powershell
py -3 scripts/query.py --id physci-001 --retrieve-only
```

List sample benchmark questions:

```powershell
py -3 scripts/query.py --list
py -3 scripts/query.py --list --type multimodal-qa
```

## Python API

```python
from physci_rag import PhySciRAG

rag = PhySciRAG()
result = rag.query(record_id="physci-001")
print(result.answer)
for ctx in result.contexts:
    print(ctx.chunk.source_file, ctx.score)
```

## Project layout

```
physci_rag/
  benchmark.py   # load PhySciBench records
  download.py    # fetch PDFs from Hugging Face
  ingest.py      # PDF text extraction + chunking
  store.py       # hybrid vector/BM25 index
  generator.py   # LLM answer generation
  pipeline.py    # high-level RAG interface
scripts/
  build_index.py
  query.py
data/
  physcibench.json
  files/         # downloaded PDFs
  index/         # built index artifacts
```

## Notes

- Image figures (`.png`/`.jpg`) are indexed with CLIP embeddings plus OCR caption text (RapidOCR).
- Rebuild the index after pulling this change: `py -3 scripts/build_index.py --skip-download` (if files are already downloaded).
- Use `--no-ocr` on `build_index.py` to skip OCR during image ingestion.
- LLM generation still receives extracted caption text for images, not raw pixels, unless you extend the generator with a vision model.
- Hybrid retrieval uses `sentence-transformers/all-MiniLM-L6-v2` for text, `sentence-transformers/clip-ViT-B-32` for images, plus BM25 reranking.
- PhySciBench is for academic research only. See the dataset card for license restrictions.

## Citation

```bibtex
@article{jiang2026physcidr,
  title   = {Deep Research in Physical Sciences: A Multi-Agent Framework and Comprehensive Benchmark},
  author  = {Jiang, Yigeng and others},
  journal = {arXiv preprint arXiv:2606.18648},
  year    = {2026}
}
```
