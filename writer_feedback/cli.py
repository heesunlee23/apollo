from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from .corpus_loader import CorpusStats, read_csv_texts
from .features import collect_ngram_counts, compute_corpus_feature_distributions
from .feedback import generate_feedback_from_corpus


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cmd_train(args: argparse.Namespace) -> None:
    texts = read_csv_texts(args.corpus, text_column=args.text_col, encoding=args.encoding)
    if not texts:
        raise SystemExit("Corpus is empty after reading the specified column.")

    # Feature distributions
    distributions = compute_corpus_feature_distributions(texts)

    # N-gram profiles
    profiles: Dict[str, List[Tuple[str, int]]] = {}
    for n in range(1, args.max_ngram + 1):
        counts = collect_ngram_counts(texts, n=n)
        top = counts.most_common(args.top_k)
        profiles[str(n)] = top

    stats = CorpusStats(
        num_documents=len(texts),
        mean_num_words=distributions["num_words"]["mean"],
        std_num_words=distributions["num_words"]["std"],
        mean_num_sentences=distributions["num_sentences"]["mean"],
        std_num_sentences=distributions["num_sentences"]["std"],
        mean_avg_sentence_length=distributions["avg_sentence_length"]["mean"],
        std_avg_sentence_length=distributions["avg_sentence_length"]["std"],
        mean_avg_word_length=distributions["avg_word_length"]["mean"],
        std_avg_word_length=distributions["avg_word_length"]["std"],
        mean_type_token_ratio=distributions["type_token_ratio"]["mean"],
        std_type_token_ratio=distributions["type_token_ratio"]["std"],
        ngram_profiles=profiles,
        metadata={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus_path": os.path.abspath(args.corpus),
            "text_column": args.text_col,
        },
    )

    _ensure_dir(args.artifacts)
    out_path = os.path.join(args.artifacts, "corpus_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(stats.to_json())
    print(f"Saved corpus stats to: {out_path}")


def _load_corpus_stats(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    return blob


def cmd_analyze(args: argparse.Namespace) -> None:
    # Load corpus stats
    stats_path = args.model if args.model else os.path.join(args.artifacts, "corpus_stats.json")
    blob = _load_corpus_stats(stats_path)

    distributions = {
        "num_words": {"mean": blob["mean_num_words"], "std": blob["std_num_words"]},
        "num_sentences": {
            "mean": blob["mean_num_sentences"],
            "std": blob["std_num_sentences"],
        },
        "avg_sentence_length": {
            "mean": blob["mean_avg_sentence_length"],
            "std": blob["std_avg_sentence_length"],
        },
        "avg_word_length": {
            "mean": blob["mean_avg_word_length"],
            "std": blob["std_avg_word_length"],
        },
        "type_token_ratio": {
            "mean": blob["mean_type_token_ratio"],
            "std": blob["std_type_token_ratio"],
        },
    }
    profiles = blob.get("ngram_profiles", {})

    # Read input
    if args.text:
        text = args.text
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise SystemExit("Provide --text or --input-file")

    report = generate_feedback_from_corpus(
        text=text,
        corpus_distributions=distributions,
        corpus_ngram_profiles=profiles,
        overuse_factor=args.overuse_factor,
        top_k_surface=args.top_k_surface,
    )

    payload = json.loads(report.to_json())
    if args.output:
        _ensure_dir(os.path.dirname(os.path.abspath(args.output)) or ".")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved analysis report to: {args.output}")
    else:
        if args.pretty:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="코퍼스 기반 쓰기 자동 피드백 시스템 (baseline)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="CSV 코퍼스에서 기준 모델(통계) 생성")
    p_train.add_argument("--corpus", required=True, help="CSV 파일 경로")
    p_train.add_argument("--text-col", default="text", help="텍스트 컬럼명")
    p_train.add_argument("--encoding", default=None, help="파일 인코딩(미지정 시 자동 시도)")
    p_train.add_argument("--artifacts", default="artifacts", help="산출물 저장 디렉토리")
    p_train.add_argument("--top-k", type=int, default=5000, help="각 n그램 상위 빈도수 저장 개수")
    p_train.add_argument("--max-ngram", type=int, default=3, help="최대 n그램 크기")
    p_train.set_defaults(func=cmd_train)

    p_analyze = sub.add_parser("analyze", help="텍스트 분석 및 피드백")
    p_analyze.add_argument("--artifacts", default="artifacts", help="산출물 디렉토리")
    p_analyze.add_argument("--model", default=None, help="직접 지정할 모델 JSON 경로")
    g_in = p_analyze.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--text", default=None, help="직접 입력 텍스트")
    g_in.add_argument("--input-file", default=None, help="분석할 텍스트 파일 경로")
    p_analyze.add_argument("--overuse-factor", type=float, default=2.0, help="구문 과다사용 배수 임계값")
    p_analyze.add_argument("--top-k-surface", type=int, default=5, help="표면화할 과다 표현 개수")
    p_analyze.add_argument("--pretty", action="store_true", help="사람이 읽기 쉬운 JSON 출력")
    p_analyze.add_argument("--output", default=None, help="JSON 결과 저장 파일 경로")
    p_analyze.set_defaults(func=cmd_analyze)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

