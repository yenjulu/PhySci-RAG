from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


def _normalize_whitespace(text: str) -> str:
  text = text.replace("\x00", " ")
  text = re.sub(r"\s+", " ", text)
  return text.strip()


@lru_cache(maxsize=1)
def _ocr_engine():
  from rapidocr_onnxruntime import RapidOCR

  return RapidOCR()


def extract_image_text(path: Path) -> str:
  """Extract visible text (captions, labels) from a scientific figure image."""
  result, _ = _ocr_engine()(str(path))
  if not result:
    return ""

  lines = [_normalize_whitespace(item[1]) for item in result if item[1].strip()]
  return _normalize_whitespace(" ".join(lines))
