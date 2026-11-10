# 프로젝트 구조 및 실험 관리 규칙

- **작성일**: 2026-11-09
- **변경 이력**
    - 2026-11-09: 최초 작성
    - 2026-11-09: 패키지 관리를 `requirements.txt`에서 uv(`pyproject.toml`, `uv.lock`, `.python-version`)로 변경, 원본 데이터셋(`dataset/`)을 저장소 밖으로 분리
    - 2026-11-09: 실험 결과 파일 형식을 CSV에서 JSON으로 통일
    - 2026-11-10: 커밋 메시지 규칙 추가
    - 2026-11-10: 실행 ID와 Qdrant Collection 이름에 임베딩 모델 코드 추가(같은 청크로 임베딩 모델을 바꿔 비교하기 위함)
    - 2026-11-10: 실험 결과를 데이터셋별 폴더로 분리하고, 보고서(`report.yaml`)와 무효 결과(`invalid/`)를 결과 옆에 두도록 변경(여러 데이터셋을 같은 코드로 실행하기 위함)
    - 2026-11-10: 임베딩·데이터셋 설정(`configs/embedding/`, `configs/datasets.yaml`), 데이터 준비 스크립트(`scripts/`) 추가
    - 2026-11-10: 모든 실험에 비용(시간·토큰) 기록 원칙 추가
    - 2026-11-10: 실행 환경 항목 추가(Qdrant 접속 주소, GPU용 PyTorch)

## 1. 디렉토리 구조

```
manufacturing-doc-rag/
├─ backend/                  # FastAPI 서비스
│  ├─ api/                   # /health, /query 라우터
│  ├─ core/                  # 설정, 로깅, 토큰 측정, 비용 기록
│  ├─ ingestion/             # PDF 추출, 청킹
│  ├─ retrieval/             # 임베딩, 색인·검색(Dense, Hybrid, Reranker), 문맥 채우기
│  ├─ generation/            # 프롬프트, LLM 호출
│  └─ db/                    # PostgreSQL 연결(구축 시)
├─ frontend/                 # Streamlit
├─ experiments/              # 실험별 설계·실행·결과
├─ evaluation/               # Hit@k, Recall@k, RAGAS, LLM-as-a-Judge
├─ configs/                  # 데이터셋·청킹·임베딩·프롬프트·검색 방식 버전별 설정
├─ scripts/                  # 데이터 준비 등 보조 스크립트
├─ infra/                    # docker-compose.yaml(Qdrant, PostgreSQL)
├─ docs/                     # 일지, 결정 사항
├─ tests/
├─ data/                     # 원본 압축 해제 데이터와 가공 데이터 (Git 제외)
├─ .env.example
├─ .python-version           # Python 버전 고정(uv)
├─ pyproject.toml            # 패키지 목록(uv)
├─ uv.lock                   # 설치 버전 고정(uv)
└─ README.md
```

- **공통 코드 공유**: 추출·청킹·임베딩·검색 코드는 `backend/`에, 평가 코드는 `evaluation/`에만 작성하고, 실험 스크립트와 API는 이 모듈을 불러 사용함.
- **backend / frontend 분리**: Streamlit은 FastAPI를 호출하는 화면이므로 독립 실행 단위로 분리함.
- **infra 분리**: Qdrant와 PostgreSQL을 docker-compose로 관리하여 README만으로 같은 환경을 재현할 수 있도록 함.
- **단계별 생성**: 각 폴더는 해당 단계에 도달했을 때 생성함.

## 2. 실험 디렉토리

```
experiments/
├─ 01_token_budget/
├─ 02_chunking/
├─ 03_prompt/
└─ 04_retrieval/
    ├─ README.md             # 설계·가설·예측·결과·결론
    ├─ config.yaml           # 고정 조건과 변수
    ├─ run.py                # 실행 스크립트
    └─ results/
        └─ {dataset}/
            ├─ {실행 ID}_{내용}.json   # 실험 결과
            ├─ report.yaml             # 보고서(목적·예측·결론), baseline 확정 시 작성
            └─ invalid/                # 무효 결과와 사유(README.md)

configs/
├─ datasets.yaml             # 데이터셋별 ZIP 위치와 기대 PDF 수
├─ chunking/                 # c1.yaml, c2.yaml
├─ embedding/                # m1.yaml
├─ prompts/                  # p1.yaml, p2.yaml
└─ retrieval/                # r1.yaml, r2.yaml, r3.yaml
```

- **번호로 순서와 의존 관계 표시**: 앞 실험에서 결정한 값이 다음 실험의 고정 조건이 됨.
- **부품 설정과 실험 설정 분리**: 데이터셋·청킹·임베딩·프롬프트·검색 방식 설정은 `configs/`에 한 번만 정의하고, 실험의 `config.yaml`은 버전 코드로 참조만 함.
- **결과 보관 기준**: 실험 결과는 JSON으로 통일하여 `results/{dataset}/`에 저장하고 커밋함. 대용량 원시 실행 결과는 `data/`와 Langfuse에 보관함.
- **실행 단위**: 한 번 실행으로 여러 데이터셋·버전을 처리하더라도, 조건마다 실행 ID와 결과 파일을 따로 만듦. 조건이 다른 결과를 하나의 실행 ID로 묶지 않음.
- **예측 먼저 기록**: 실험 README의 가설과 예측은 실행 전에 작성하고, 결과를 본 뒤 고치지 않음.
- **보고서와 baseline**: 결과를 검토해 기준값으로 쓰기로 결정한 데이터셋만 `report.yaml`을 작성하며, 보고서 화면에는 `report.yaml`이 있는 데이터셋만 표시함.
- **무효 결과 보관**: 잘못된 결과는 지우지 않고 `invalid/`로 옮긴 뒤, `invalid/README.md`에 대상·사유·무효 범위·조치를 기록함.
- **비용 기록**: 모든 실험은 결과 JSON의 `summary.cost`에 같은 구조로 단계별 시간, 요청별 시간, 토큰 수, 실행 환경을 기록함. 확인하지 못한 값은 0이 아니라 `null`로 남김.

## 3. 명명 규칙

- **공통**: 영문 소문자를 사용하고 단어는 `_`로 구분함. 한글과 공백은 사용하지 않음.
    - 외부 도구가 관례로 정한 파일 이름은 그대로 따름(예: `docker-compose.yaml`).
    - YAML 파일의 확장자는 `.yaml`로 통일함.
- **버전 코드**: 한 번 사용한 코드는 재사용하지 않고, 설정이 바뀌면 번호를 올림.
- **미사용 조건**: 실험에서 사용하지 않는 조건은 `p0`, `r0`처럼 0으로 표기함.

| 대상 | 코드 | 설정 파일 |
|---|---|---|
| 데이터셋 | `manufacturing_basic`, `manufacturing_intermediate`, `manufacturing_advanced` | `configs/datasets.yaml` |
| 청킹 | `c1`, `c2`, `c3` | `configs/chunking/c1.yaml` |
| 임베딩 | `m1`(`intfloat/multilingual-e5-base`) | `configs/embedding/m1.yaml` |
| 프롬프트 | `p1`, `p2` | `configs/prompts/p1.yaml` |
| 검색 방식 | `r1`(Naive), `r2`(Hybrid), `r3`(Hybrid+Reranker) | `configs/retrieval/r1.yaml` |

| 대상 | 형식 | 예시 |
|---|---|---|
| 실험 폴더 | `{번호2자리}_{주제}` | `02_chunking` |
| 실행 ID | `e{실험번호}_{청킹}_{임베딩}_{프롬프트}_{검색}_{YYYYMMDD}_{HHMM}` | `e02_c2_m1_p0_r1_20261110_1926` |
| 결과 파일 | `results/{dataset}/{실행 ID}_{내용}.json` | `results/manufacturing_basic/e02_c2_m1_p0_r1_20261110_1926_retrieval.json` |
| Qdrant Collection | `{데이터셋}_{청킹}_{임베딩}` | `manufacturing_basic_c2_m1` |

- 임베딩 코드를 추가하기 전에 실행한 결과(01 토큰 한도 실험, 02 청크 생성 단계)는 실행 ID에 임베딩 자리가 없음(예: `e01_c0_p0_r0_…`).

## 4. Langfuse 관리

| 프로젝트 규칙 | Langfuse |
|---|---|
| 요청 하나 | trace, `request_id` = trace ID |
| 실행 ID | trace의 tag와 metadata |
| 버전 코드(`c`, `m`, `p`, `r`), 질문 ID | trace metadata에 각각 저장 |
| 요청 출처 | trace metadata `source`(`service` 또는 `evaluation`) |
| Hit@k, Recall@k, Faithfulness, Judge 네 항목 | trace에 연결하는 score |
| 평가 질문 묶음 | Dataset |
| 실험 실행 | Dataset run(이름 = 실행 ID) |

- **버전 코드 별도 저장**: 실행 ID 문자열을 분해하지 않고 필터링할 수 있도록 버전 코드를 metadata에 각각 저장함.
- **요청 출처 구분**: 서비스 질문과 평가 실행 기록이 섞이지 않도록 `source`로 구분하며, 관리자 대시보드는 `source: service`만 표시함.
- **Dataset run 활용**: 같은 질문에 대한 실행 ID별 결과를 Langfuse 화면에서 비교함.
- **적용 범위**: LLM 호출이 없는 토큰 한도 결정 실험은 기록 대상에서 제외하고, 청킹 실험부터 적용함.

## 5. 커밋 메시지

- **형식**: `{type}({scope}): {요약}`. type과 scope는 영어, 요약과 본문은 한국어로 작성함.
- **type**: `feat`(기능 추가), `fix`(버그 수정), `docs`(문서), `refactor`(동작 변화 없는 구조 변경), `chore`(설정·의존성), `exp`(실험 실행·결과)
- **scope**: 바뀐 영역(`core`, `ingestion`, `frontend`, `token_budget`, `chunking` 등)
- **요약**: 무엇을 왜 했는지 한 줄로 작성함. 세부 내용은 본문에 작성함.
- **예**: `fix(core): 생성 토크나이저 근사 여부 기록 오류 수정 및 토큰 한도 재실행`

## 6. 실행 환경

- **Qdrant**: `infra/docker-compose.yaml`로 실행하며, 포트는 이 컴퓨터에서만 접속하도록 `127.0.0.1`에 엶.
- **Qdrant 접속 주소**: 코드에서는 `http://127.0.0.1:6333`을 사용함. Windows에서 `localhost`는 IPv6(`::1`)를 먼저 시도하므로, IPv4에만 연 포트에 접속할 때 요청마다 약 2초가 지연됨.
- **GPU용 PyTorch**: Windows에서 PyPI의 기본 PyTorch는 CPU 전용이므로, `pyproject.toml`에 PyTorch CUDA 인덱스(`cu128`)를 지정해 설치함. RTX 50 시리즈는 CUDA 12.8 이상으로 빌드된 PyTorch가 필요함.
