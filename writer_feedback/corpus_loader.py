from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple


@dataclass
class CorpusStats:
    """Aggregated statistics computed from a corpus.

    The numeric fields are simple distribution descriptors used to compare a new
    text sample against the corpus. All values are computed across documents.
    """

    num_documents: int
    mean_num_words: float
    std_num_words: float
    mean_num_sentences: float
    std_num_sentences: float
    mean_avg_sentence_length: float
    std_avg_sentence_length: float
    mean_avg_word_length: float
    std_avg_word_length: float
    mean_type_token_ratio: float
    std_type_token_ratio: float

    # N-gram profiles: mapping n -> list of (ngram, count)
    ngram_profiles: Dict[str, List[Tuple[str, int]]]

    # Metadata for traceability
    metadata: Dict[str, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def read_csv_texts(
    csv_path: str,
    text_column: str = "text",
    encoding: str | None = None,
) -> List[str]:
    """Read a CSV file and return a list of text entries from a specific column.

    Empty or whitespace-only rows are ignored.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV path not found: {csv_path}")

    # Try utf-8 by default, fall back to utf-8-sig if needed
    encodings_to_try: List[str] = [encoding] if encoding else ["utf-8", "utf-8-sig"]
    last_error: Exception | None = None
    for enc in encodings_to_try:
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                if text_column not in (reader.fieldnames or []):
                    raise KeyError(
                        f"Column '{text_column}' not found. Available: {reader.fieldnames}"
                    )
                texts: List[str] = []
                for row in reader:
                    value = (row.get(text_column) or "").strip()
                    if value:
                        texts.append(value)
                return texts
        except Exception as e:  # noqa: BLE001 - surface read errors clearly
            last_error = e
            continue

    assert last_error is not None
    raise last_error


