from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

from .features import TextFeatures, collect_ngram_counts, extract_features, tokenize_words


@dataclass
class FeedbackItem:
    metric: str
    value: float
    corpus_mean: float | None
    corpus_std: float | None
    severity: str  # info | suggestion | warning
    message: str
    action: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class FeedbackReport:
    features: TextFeatures
    items: List[FeedbackItem]

    def to_json(self) -> str:
        payload = {
            "features": self.features.to_dict(),
            "items": [i.to_dict() for i in self.items],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _bounded_label(delta_std: float) -> str:
    if abs(delta_std) < 0.5:
        return "info"
    if abs(delta_std) < 1.5:
        return "suggestion"
    return "warning"


def generate_feedback_from_corpus(
    text: str,
    corpus_distributions: Dict[str, Dict[str, float]],
    corpus_ngram_profiles: Dict[str, List[Tuple[str, int]]],
    overuse_factor: float = 2.0,
    top_k_surface: int = 5,
) -> FeedbackReport:
    features = extract_features(text)
    feats = features.to_dict()
    items: List[FeedbackItem] = []

    # Compare scalar features to corpus norms
    for name, stats in corpus_distributions.items():
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 1e-9)) or 1e-9
        value = float(feats[name])
        delta_std = (value - mean) / std
        level = _bounded_label(delta_std)

        if name == "avg_sentence_length" and value > mean + std:
            message = "문장 길이가 평균보다 깁니다. 가독성을 위해 긴 문장을 나눠보세요."
            action = "접속사/쉼표 기준으로 두 개 이상의 짧은 문장으로 분리"
        elif name == "avg_sentence_length" and value < mean - std:
            message = "문장 길이가 평균보다 짧습니다. 내용 연결이 부자연스러울 수 있어요."
            action = "문장 간 연결어 추가 및 세부 설명 보강"
        elif name == "type_token_ratio" and value < mean - std:
            message = "어휘 다양성이 낮습니다. 동일 어휘 반복 사용을 줄여보세요."
            action = "동의어 사용, 내용 전개 시 구체적 표현 추가"
        elif name == "avg_word_length" and value > mean + std:
            message = "단어 길이가 길어 전문용어/합성어 비중이 높을 수 있습니다."
            action = "필요 시 쉬운 표현으로 대체하거나 간단한 정의를 병기"
        elif name == "num_sentences" and value == 1:
            message = "한 문장으로만 구성되었습니다. 구조화가 필요할 수 있어요."
            action = "주제별로 문장을 구분하고 단락을 구성"
        else:
            # Generic note; keep minimal to avoid noise
            message = f"{name} 값이 코퍼스 평균과 {abs(delta_std):.1f}σ 차이입니다."
            action = "맥락에 맞게 해당 지표 균형화 시도"

        items.append(
            FeedbackItem(
                metric=name,
                value=value,
                corpus_mean=mean,
                corpus_std=std,
                severity=level,
                message=message,
                action=action,
            )
        )

    # Phrase overuse detection via n-grams
    tokens = tokenize_words(text)
    unigram_counts = collect_ngram_counts([text], n=1)
    bigram_counts = collect_ngram_counts([text], n=2)
    trigram_counts = collect_ngram_counts([text], n=3)

    def to_rate(counter, denom):
        return {k: v / max(1, denom) for k, v in counter.items()}

    doc_rates = {
        "1": to_rate(unigram_counts, len(tokens)),
        "2": to_rate(bigram_counts, max(1, len(tokens) - 1)),
        "3": to_rate(trigram_counts, max(1, len(tokens) - 2)),
    }

    for n in ["2", "3"]:
        corpus_profile = dict(corpus_ngram_profiles.get(n, [])[: 10000])
        if not corpus_profile:
            continue
        # Normalize corpus counts to rates
        corpus_total = sum(corpus_profile.values()) or 1
        corpus_rates = {k: v / corpus_total for k, v in corpus_profile.items()}

        overused: List[Tuple[str, float]] = []
        for phrase, rate in doc_rates[n].items():
            base = corpus_rates.get(phrase, 1e-9)
            if rate > base * overuse_factor and rate > 0.0:
                overused.append((phrase, rate / base))

        if overused:
            overused.sort(key=lambda x: x[1], reverse=True)
            sample = ", ".join(p for p, _ in overused[:top_k_surface])
            items.append(
                FeedbackItem(
                    metric=f"overused_{n}gram_phrases",
                    value=float(len(overused)),
                    corpus_mean=None,
                    corpus_std=None,
                    severity="suggestion",
                    message=f"반복적으로 사용하는 {n}-그램 표현이 많습니다: {sample}",
                    action="반복 구문을 간결화하거나 동의 표현으로 변주",
                )
            )

    return FeedbackReport(features=features, items=items)

