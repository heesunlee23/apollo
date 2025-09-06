# 코퍼스 기반 쓰기 자동 피드백 시스템 (Python)

이 패키지는 CSV 코퍼스를 바탕으로 문서 지표 분포와 n-그램 프로필을 학습하고, 새로운 텍스트를 분석하여 코퍼스 기준과의 차이를 바탕으로 자동 피드백을 제공합니다.

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 코퍼스 형식

- CSV 파일의 한 컬럼에 텍스트가 존재해야 합니다.
- 기본 컬럼명은 `text`이며 `--text-col`로 변경 가능합니다.

예시(`examples/corpus.csv`):

```csv
text
이것은 간단한 예시 문장입니다. 두 번째 문장도 포함되어 있습니다.
짧은 문장입니다!
이 텍스트는 어휘 다양성을 테스트하기 위한 텍스트입니다. 텍스트가 반복됩니다.
```

## 사용법

1) 기준 통계 학습

```bash
python -m writer_feedback.cli train --corpus examples/corpus.csv --artifacts artifacts
```

2) 텍스트 분석

```bash
python -m writer_feedback.cli analyze --artifacts artifacts --text "분석할 문장을 여기에 입력합니다. 길이가 너무 길다면 문장을 나눠 보세요."
```

또는 파일 입력:

```bash
python -m writer_feedback.cli analyze --artifacts artifacts --input-file examples/input.txt --pretty
```

## 주요 지표

- 평균 문장 길이(avg_sentence_length), 평균 단어 길이(avg_word_length)
- 어휘 다양성(type_token_ratio), 문장 수(num_sentences), 단어 수(num_words)
- 2/3-그램 과다 사용 감지

## 참고

- 단순 정규식 토크나이저를 사용합니다. 정밀 분석이 필요하면 형태소 분석기나 언어 모델로 확장하세요.
