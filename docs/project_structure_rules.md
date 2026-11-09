프로젝트 구조 및 실험 관리 규칙
작성일: 2026-11-09
변경 이력
2026-11-09: 최초 작성
2026-11-09: 패키지 관리를 requirements.txt에서 uv(pyproject.toml, uv.lock, .python-version)로 변경, 원본 데이터셋(dataset/)을 저장소 밖으로 분리
1. 디렉토리 구조
manufacturing-doc-rag/
├─ backend/                  # FastAPI 서비스
│  ├─ api/                   # /health, /query 라우터
│  ├─ core/                  # 설정, 로깅
│  ├─ ingestion/             # PDF 추출, 청킹, 색인
│  ├─ retrieval/             # Dense, Hybrid, Reranker
│  ├─ generation/            # 프롬프트, LLM 호출
│  └─ db/                    # Qdrant·PostgreSQL 연결(구축 시)
├─ frontend/                 # Streamlit
├─ experiments/              # 실험별 설계·실행·결과
├─ evaluation/               # Hit@k, Recall@k, RAGAS, LLM-as-a-Judge
├─ configs/                  # 청킹·프롬프트·검색 방식 버전별 설정
├─ infra/                    # docker-compose(Qdrant, PostgreSQL)
├─ docs/                     # 일지, 결정 사항
├─ tests/
├─ data/                     # 가공 데이터 (Git 제외)
├─ .env.example
├─ .python-version           # Python 버전 고정(uv)
├─ pyproject.toml            # 패키지 목록(uv)
├─ uv.lock                   # 설치 버전 고정(uv)
└─ README.md
추출·청킹 코드 공유: 추출·청킹 코드는 backend/ingestion/에만 작성하고, 실험 스크립트는 이 모듈을 불러 사용함.
backend / frontend 분리: Streamlit은 FastAPI를 호출하는 화면이므로 독립 실행 단위로 분리함.
infra 분리: Qdrant와 PostgreSQL을 docker-compose로 관리하여 README만으로 같은 환경을 재현할 수 있도록 함.
단계별 생성: 각 폴더는 해당 단계에 도달했을 때 생성함.
2. 실험 디렉토리
experiments/
├─ 01_token_budget/
├─ 02_chunking/
├─ 03_prompt/
└─ 04_retrieval/
    ├─ README.md             # 설계·가설·예측·결과·결론
    ├─ config.yaml           # 고정 조건과 변수
    ├─ run.py                # 실행 스크립트
    └─ results/              # 요약 결과(CSV, 표)

configs/
├─ chunking/                 # c1.yaml, c2.yaml
├─ prompts/                  # p1.yaml, p2.yaml
└─ retrieval/                # r1.yaml, r2.yaml, r3.yaml
번호로 순서와 의존 관계 표시: 앞 실험에서 결정한 값이 다음 실험의 고정 조건이 됨.
부품 설정과 실험 설정 분리: 청킹·프롬프트·검색 방식 설정은 configs/에 한 번만 정의하고, 실험의 config.yaml은 버전 코드로 참조만 함.
결과 보관 기준: 원시 실행 결과는 data/와 Langfuse에 보관하고, 저장소에는 요약 결과만 커밋함.
3. 명명 규칙
공통: 영문 소문자를 사용하고 단어는 _로 구분함. 한글과 공백은 사용하지 않음.
버전 코드: 한 번 사용한 코드는 재사용하지 않고, 설정이 바뀌면 번호를 올림.
미사용 조건: 실험에서 사용하지 않는 조건은 p0, r0처럼 0으로 표기함.
대상	코드	설정 파일
청킹	c1, c2	configs/chunking/c1.yaml
프롬프트	p1, p2	configs/prompts/p1.yaml
검색 방식	r1(Naive), r2(Hybrid), r3(Hybrid+Reranker)	configs/retrieval/r1.yaml
대상	형식	예시
실험 폴더	{번호2자리}_{주제}	02_chunking
실행 ID	e{실험번호}_{청킹}_{프롬프트}_{검색}_{YYYYMMDD}_{HHMM}	e02_c2_p1_r1_20261110_1430
결과 파일	{실행 ID}_{내용}.csv	e02_c2_p1_r1_20261110_1430_retrieval.csv
Qdrant Collection	{데이터셋}_{청킹}	manufacturing_basic_c2
4. Langfuse 관리
프로젝트 규칙	Langfuse
요청 하나	trace, request_id = trace ID
실행 ID	trace의 tag와 metadata
버전 코드(c, p, r), 질문 ID	trace metadata에 각각 저장
Hit@k, Recall@k, Faithfulness, Judge 네 항목	trace에 연결하는 score
평가 질문 묶음	Dataset
실험 실행	Dataset run(이름 = 실행 ID)
버전 코드 별도 저장: 실행 ID 문자열을 분해하지 않고 필터링할 수 있도록 버전 코드를 metadata에 각각 저장함.
Dataset run 활용: 같은 질문에 대한 실행 ID별 결과를 Langfuse 화면에서 비교함.
적용 범위: LLM 호출이 없는 토큰 한도 결정 실험은 기록 대상에서 제외하고, 청킹 실험부터 적용함.