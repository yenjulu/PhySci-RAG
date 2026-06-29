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
   PDF chunking (pypdf) + image indexing (CLIP + OCR)
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

Create and activate a Conda environment, then install Python dependencies:

```powershell
cd PhySci-RAG
conda env create -f environment.yml
conda activate physci-rag
pip install -r requirements.txt
```

To recreate the environment later:

```powershell
conda env remove -n physci-rag
conda env create -f environment.yml
conda activate physci-rag
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you want generated answers. Without it, `query.py` still returns retrieved passages.

The benchmark JSON is already included in `data/physcibench.json`.

## Build the index

Download source PDFs/images and build the vector index:

```powershell
# Quick test with the first 10 source files (~few minutes)
python scripts/build_index.py --download-limit 10

# Full index (157 PDFs + 18 images — takes longer)
python scripts/build_index.py

# PDF text only (skip image figures)
python scripts/build_index.py --pdf-only
```

Re-run ingestion only (if source files are already downloaded):

```powershell
python scripts/build_index.py --skip-download
```

## Query

Ask a free-form question:

```powershell
python scripts/query.py "What dopants were studied in CsV3Sb5 STM experiments?"
```

Run against a specific benchmark item (uses its referenced files as a retrieval filter):

```powershell
python scripts/query.py --id physci-001
```

Retrieval only (no LLM):

```powershell
python scripts/query.py --id physci-001 --retrieve-only
```

List sample benchmark questions:

```powershell
python scripts/query.py --list
python scripts/query.py --list --type multimodal-qa
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
  download.py    # fetch PDFs/images from Hugging Face
  ingest.py      # PDF text extraction, image chunking
  ocr.py         # OCR (Optical Character Recognition) for figure captions
  store.py       # hybrid vector/BM25 index
  generator.py   # LLM answer generation
  pipeline.py    # high-level RAG interface
scripts/
  build_index.py
  query.py
environment.yml  # Conda environment definition
data/
  physcibench.json
  files/         # downloaded PDFs and images
  index/         # built index artifacts
```

## Image text extraction (OCR)

Benchmark figure images (`.png`, `.jpg`) are not read as pixels by the LLM. Instead, the pipeline uses **OCR (Optical Character Recognition)** to extract visible text such as figure captions, axis labels, and panel markers during indexing.

- **Implementation:** [RapidOCR](https://github.com/RapidAI/RapidOCR) via `rapidocr-onnxruntime`
- **Module:** `physci_rag/ocr.py`
- **What OCR captures:** printed caption and label text at indexing time
- **What OCR does not capture:** chart curves, STM topography patterns, or other purely visual content
- **Retrieval:** OCR text improves BM25/keyword matching; CLIP embeddings still handle visual similarity

Disable OCR at build time with `--no-ocr` if needed.

## Notes

- Rebuild the index after pulling OCR changes: `python scripts/build_index.py --skip-download` (if files are already downloaded).
- LLM generation receives OCR-extracted caption text for images, not raw pixels, unless you extend the generator with a vision model.
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
