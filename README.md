# AI Pack - Semantic Chunking with BGE-M3

BGE-M3 임베딩 모델을 활용한 시맨틱 청킹 도구입니다.  
다양한 형식의 문서를 마크다운으로 변환하고, 의미 기반으로 분할하며, 헤딩 계층 구조를 보존합니다.

## 특징

- **다양한 문서 형식 지원**: Microsoft markitdown 활용
  - 오피스 문서: Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
  - PDF, HTML, XML, JSON, CSV
  - 이미지 (EXIF/OCR), 오디오 (음성 인식), 비디오 (자막 추출)
  - 코드 파일, Jupyter Notebook, ZIP 아카이브
- **Azure AI 서비스 연동** (선택사항)
  - Document Intelligence: 스캔 PDF, 이미지 OCR 향상
  - Azure OpenAI (GPT-4o): 이미지 내용 이해
  - 설정된 서비스만 자동 활성화
- **시맨틱 청킹**: 의미 유사도 기반 청크 분할
- **마크다운 구조 보존**: 헤딩 레벨, 섹션 경로 등 계층 정보 유지
- **다국어 지원**: BGE-M3의 100+ 언어 지원
- **증분 업데이트**: 콘텐츠 해시 기반 변경 감지
- **zstd 압축**: 효율적인 parquet 저장

## 지원 파일 형식

| 카테고리 | 확장자 |
| ------ | ------ |
| 오피스 문서 | `.docx`, `.doc`, `.xlsx`, `.xls`, `.pptx`, `.ppt` |
| PDF/웹 | `.pdf`, `.html`, `.htm`, `.xml`, `.json`, `.csv` |
| 마크다운/텍스트 | `.md`, `.markdown`, `.txt`, `.rst` |
| 이미지 (EXIF/OCR) | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`, `.tiff` |
| 오디오 (음성 인식) | `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac` |
| 비디오 (자막 추출) | `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm` |
| 코드/기타 | `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.ipynb`, `.zip` |

## 모듈 구성

| 모듈 | 설명 |
| ------ | ------ |
| `01_download_model.py` | BGE-M3 임베딩 모델 다운로드 |
| `02_prepare_content.py` | 메타데이터 추출 및 YAML front matter 생성 |
| `03_semantic_chunking.py` | 시맨틱 청킹 및 parquet 저장 |

## 설치

```bash
# uv 사용 (권장)
uv sync

# 또는 pip
pip install FlagEmbedding mistune pyarrow pandas pyyaml markitdown[all]
```

## Azure 서비스 연동 (선택사항)

기본 markitdown만으로도 동작하지만, Azure 서비스를 연동하면 더 나은 결과를 얻을 수 있습니다.

### 지원 서비스

| 서비스 | 용도 | 향상되는 기능 |
| ------ | ------ | ------ |
| Document Intelligence | 스캔 PDF, 이미지 OCR | 텍스트 추출 정확도 |
| Azure OpenAI (GPT-4o) | 이미지 내용 이해 | 이미지 설명 생성 |

### 설정 방법

```bash
# 1. 환경 파일 생성
cp .env.example .env

# 2. 필요한 키만 입력 (설정된 서비스만 활성화됨)
```

`.env` 파일 예시:

```env
# Document Intelligence (스캔 PDF, 이미지 OCR)
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key

# Azure OpenAI (이미지 내용 이해)
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
```

### 동작 방식

- **키 없음**: 기본 markitdown만 사용
- **일부 서비스만 설정**: 해당 서비스만 활성화
- **모두 설정**: 전체 기능 활성화

실행 시 연동 상태가 표시됩니다:

```text
🔗 Azure 서비스 연동: Document Intelligence, OpenAI (gpt-4o)
```

또는:

```text
ℹ️ Azure 서비스 미연동 (기본 markitdown 사용)
```

## 사용법

### 1. 모델 다운로드 (최초 1회)

```bash
python 01_download_model.py
```

### 2. 문서 준비

`input_docs/` 디렉터리에 파일을 넣고 (다양한 형식 지원):

```bash
python 02_prepare_content.py
```

지원되는 모든 형식의 파일이 마크다운으로 변환되고 메타데이터가 추가됩니다.

### 3. 시맨틱 청킹

```bash
python 03_semantic_chunking.py
```

옵션:

- `--input-dir`: 입력 디렉터리 (기본: `prepared_contents`)
- `--output-dir`: 출력 디렉터리 (기본: `chunked_data`)
- `--similarity-threshold`: 유사도 임계값 (기본: 0.5)

## 출력 스키마

| 필드 | 타입 | 설명 |
| ------ | ------ | ------ |
| `chunk_id` | string | 청크 고유 ID |
| `content_hash` | string | 콘텐츠 해시 (증분 업데이트용) |
| `chunk_text` | string | 청크 텍스트 |
| `chunk_type` | string | 타입 (header, paragraph, list, code, table) |
| `heading_level` | int32 | 헤딩 레벨 (0=일반, 1-6=H1-H6) |
| `heading_text` | string | 현재 헤딩 텍스트 |
| `parent_heading` | string | 부모 헤딩 텍스트 |
| `section_path` | list[string] | 섹션 계층 경로 배열 |
| `table_headers` | list[string] | 표 컬럼 헤더 (표인 경우) |
| `table_row_count` | int32 | 표 데이터 행 수 (표인 경우) |
| `domain` | string | 도메인 (메타데이터) |
| `keywords` | string | 키워드 JSON (메타데이터) |
| `version` | int32 | 버전 번호 |

## 디렉터리 구조

```text
aipack/
├── 01_download_model.py       # BGE-M3 모델 다운로드
├── 02_prepare_content.py      # 메타데이터 추출 + Azure 연동
├── 03_semantic_chunking.py    # 시맨틱 청킹
├── input_docs/                # 입력 문서
├── prepared_contents/         # 메타데이터 추가된 문서
├── chunked_data/              # 청킹된 parquet 파일
├── .env.example               # 환경 변수 템플릿
├── pyproject.toml
└── README.md
```

## 예시

### 입력 마크다운

```markdown
# 제목

## 섹션 1
내용...

## 섹션 2
다른 내용...
```

### 출력 parquet 예시

| chunk_id | heading_level | heading_text | section_path |
| ---------- | --------------- | -------------- | -------------- |
| abc123 | 1 | 제목 | # 제목 |
| def456 | 2 | 섹션 1 | # 제목 / ## 섹션 1 |
| ghi789 | 2 | 섹션 2 | # 제목 / ## 섹션 2 |

## 시스템 요구사항

- Python 3.11 이상
- 약 3GB 디스크 공간 (BGE-M3 모델용)
- 8GB 이상 RAM 권장

## 라이선스

MIT License
