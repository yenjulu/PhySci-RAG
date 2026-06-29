import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

from .benchmark import load_benchmark
from .config import FILES_DIR, HF_DATASET_REPO


def list_required_files(pdf_only: bool = False) -> list[str]:
    records = load_benchmark()
    required: set[str] = set()
    for record in records:
        required.update(record.files)

    files = sorted(required)
    if pdf_only:
        files = [name for name in files if name.lower().endswith(".pdf")]
    return files


def list_remote_files() -> list[str]:
    return [
        path
        for path in list_repo_files(HF_DATASET_REPO, repo_type="dataset")
        if path.startswith("files/")
    ]


def download_files(
    dest_dir: Path = FILES_DIR,
    only: set[str] | None = None,
    limit: int | None = None,
    pdf_only: bool = True,
) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    required = sorted(only) if only else list_required_files(pdf_only=pdf_only)
    if limit is not None:
        required = required[:limit]

    downloaded: list[Path] = []
    for filename in tqdm(required, desc="Downloading source files"):
        local_path = dest_dir / filename
        if local_path.exists():
            downloaded.append(local_path)
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=f"files/{filename}",
            repo_type="dataset",
        )
        shutil.copy2(cached, local_path)
        downloaded.append(local_path)

    return downloaded
