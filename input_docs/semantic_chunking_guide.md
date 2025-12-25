# BGE-M3 시맨틱 청킹 가이드

이 문서는 BGE-M3 임베딩 모델을 사용한 시맨틱 청킹 시스템에 대해 설명합니다.

## 개요

시맨틱 청킹은 텍스트를 의미적으로 유사한 부분끼리 그룹화하는 기술입니다. 
기존의 고정 크기 청킹과 달리, **의미 기반 경계**에서 분할하여 
더 자연스러운 컨텍스트를 유지합니다.

## 핵심 기술

### BGE-M3 임베딩 모델

BGE-M3는 다음과 같은 특징을 가진 **다국어 임베딩 모델**입니다:

- 100개 이상의 언어 지원
- 1024차원 밀집 벡터
- Dense, Sparse, ColBERT 세 가지 표현 방식

### 마크다운 파싱

mistune 라이브러리를 사용하여 마크다운 구조를 파싱합니다:

```python
import mistune

md = mistune.create_markdown(renderer=None)
tokens = md(content)
```

## 청킹 알고리즘

1. 마크다운을 섹션별로 파싱
2. 각 섹션에 대해 임베딩 계산
3. 연속 섹션 간 유사도 측정
4. 임계값 이하인 지점에서 분할

## 사용 방법

### 설치

```bash
uv add FlagEmbedding mistune pyarrow pandas
```

### 실행

```bash
python 03_semantic_chunking.py --input-dir input_docs --output-dir chunked_data
```

## 스키마 구조

출력 parquet 파일은 다음 스키마를 가집니다:

| 필드명 | 타입 | 설명 |
|--------|------|------|
| chunk_id | string | 청크 고유 ID |
| heading_level | int32 | 헤딩 레벨 (0-6) |
| section_path | list[string] | 섹션 계층 경로 배열 |
| chunk_text | string | 청크 텍스트 |
| table_headers | list[string] | 표 컬럼 헤더 (표인 경우) |
| table_row_count | int32 | 표 행 수 (표인 경우) |

### 성능 비교표

| 방식 | 처리 속도 | 품질 | GPU 필요 |
|------|-----------|------|----------|
| LLM 기반 | 느림 (분 단위) | 매우 높음 | 권장 |
| 규칙 기반 | 매우 빠름 | 낮음 | 불필요 |
| 시맨틱 (BGE-M3) | 빠름 (초 단위) | 높음 | 선택적 |

## 결론

BGE-M3 기반 시맨틱 청킹은 LLM 기반 방식보다 **10-100배 빠르면서** 
충분한 품질의 청크를 생성합니다.
