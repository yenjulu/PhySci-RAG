from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FILES_DIR = DATA_DIR / "files"
INDEX_DIR = DATA_DIR / "index"
BENCHMARK_JSON = DATA_DIR / "physcibench.json"

HF_DATASET_REPO = "yigengx/PhySciBench"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 6
HYBRID_ALPHA = 0.65  # weight for dense vs sparse (BM25)

LLM_MODEL = "gpt-4o-mini"
MAX_CONTEXT_CHARS = 12_000
