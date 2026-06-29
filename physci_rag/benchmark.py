import json
from dataclasses import dataclass
from pathlib import Path

from .config import BENCHMARK_JSON


@dataclass(frozen=True)
class BenchmarkRecord:
    id: str
    question: str
    answer: str
    category: str
    type: str
    files: list[str]
    rubrics: str | None


def load_benchmark(path: Path = BENCHMARK_JSON) -> list[BenchmarkRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records: list[BenchmarkRecord] = []
    for item in raw:
        records.append(
            BenchmarkRecord(
                id=item["id"],
                question=item["question"],
                answer=item["answer"],
                category=item["category"],
                type=item["type"],
                files=item.get("files") or [],
                rubrics=item.get("rubrics"),
            )
        )
    return records


def file_to_record_ids(records: list[BenchmarkRecord]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for record in records:
        for filename in record.files:
            mapping.setdefault(filename, []).append(record.id)
    return mapping
