from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


# Basic regex-based tokenization that works reasonably for Korean and English.
_WORD_PATTERN = re.compile(r"[A-Za-z가-힣0-9]+", flags=re.UNICODE)
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?…]+|[。！？]+|\n+", flags=re.UNICODE)


@dataclass
class TextFeatures:
    num_characters: int
    num_words: int
    num_sentences: int
    avg_sentence_length: float
    avg_word_length: float
    type_token_ratio: float
    repeated_unigrams_ratio: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "num_characters": self.num_characters,
            "num_words": self.num_words,
            "num_sentences": self.num_sentences,
            "avg_sentence_length": self.avg_sentence_length,
            "avg_word_length": self.avg_word_length,
            "type_token_ratio": self.type_token_ratio,
            "repeated_unigrams_ratio": self.repeated_unigrams_ratio,
        }


def tokenize_words(text: str) -> List[str]:
    # Lowercase for English; for Korean it generally has no effect
    return [m.group(0).lower() for m in _WORD_PATTERN.finditer(text)]


def split_sentences(text: str) -> List[str]:
    # Split on punctuation and newlines; filter empties
    parts = _SENTENCE_SPLIT_PATTERN.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def extract_features(text: str) -> TextFeatures:
    words = tokenize_words(text)
    sentences = split_sentences(text)

    num_characters = len(text)
    num_words = len(words)
    num_sentences = max(1, len(sentences))  # avoid division by zero

    # Words per sentence
    sentence_lengths = []
    if sentences:
        # Naive approximation: split each sentence with same tokenizer
        cursor = 0
        # We cannot easily map words to sentences without a tokenizer per sentence;
        # approximate using total words / sentences
        avg_sentence_length = num_words / num_sentences
    else:
        avg_sentence_length = float(num_words)

    # Average word length in characters
    avg_word_length = (sum(len(w) for w in words) / num_words) if num_words else 0.0

    # Type-token ratio
    unique_words = set(words)
    type_token_ratio = (len(unique_words) / num_words) if num_words else 0.0

    # Repetition: share of tokens that are part of repeated unigrams (frequency >= 3)
    counts = Counter(words)
    repeated_tokens = sum(freq for w, freq in counts.items() if freq >= 3)
    repeated_unigrams_ratio = (repeated_tokens / num_words) if num_words else 0.0

    return TextFeatures(
        num_characters=num_characters,
        num_words=num_words,
        num_sentences=num_sentences,
        avg_sentence_length=avg_sentence_length,
        avg_word_length=avg_word_length,
        type_token_ratio=type_token_ratio,
        repeated_unigrams_ratio=repeated_unigrams_ratio,
    )


def collect_ngram_counts(texts: Iterable[str], n: int) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        tokens = tokenize_words(text)
        if n == 1:
            counter.update(tokens)
        else:
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i : i + n])
                counter[ngram] += 1
    return counter


def compute_corpus_feature_distributions(texts: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute mean and std per feature across documents.

    Returns a mapping: feature_name -> {"mean": float, "std": float}
    """
    values: Dict[str, List[float]] = {}
    feature_names = [
        "num_words",
        "num_sentences",
        "avg_sentence_length",
        "avg_word_length",
        "type_token_ratio",
    ]

    for text in texts:
        feats = extract_features(text).to_dict()
        for name in feature_names:
            values.setdefault(name, []).append(float(feats[name]))

    distributions: Dict[str, Dict[str, float]] = {}
    for name, arr in values.items():
        mean = sum(arr) / len(arr) if arr else 0.0
        var = sum((x - mean) ** 2 for x in arr) / len(arr) if arr else 0.0
        std = math.sqrt(var)
        distributions[name] = {"mean": mean, "std": std}
    return distributions

