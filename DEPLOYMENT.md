# Streamlit Community Cloud 배포

## 배포 구조

```text
Streamlit Community Cloud
  └─ 외부 FastAPI
       ├─ Qdrant Cloud
       ├─ 외부 LLM API
       └─ Langfuse
```

## 1. Qdrant Collection 이전

로컬 Qdrant 실행 후 대상 Qdrant의 URL과 API Key 설정

```powershell
$env:QDRANT_TARGET_URL = "https://<cluster>.cloud.qdrant.io:6333"
$env:QDRANT_TARGET_API_KEY = "<qdrant-api-key>"
uv run python scripts/copy_qdrant_collections.py
```

기본 이전 대상

- `manufacturing_basic_c1_m1`
- `manufacturing_basic_c2_m1`
- `manufacturing_basic_c3_m1`

기존 대상 Collection을 교체하는 경우에만 `--recreate` 추가

## 2. FastAPI 배포

`Dockerfile.api`를 지원하는 컨테이너 호스팅 서비스에 GitHub 저장소 연결

필수 환경 변수

```text
OPENAI_API_KEY=<openai-api-key>
QDRANT_URL=https://<cluster>.cloud.qdrant.io:6333
QDRANT_API_KEY=<qdrant-api-key>
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
RAG_ACTIVE_CONFIG_PATH=configs/active.yaml
```

선택 환경 변수

```text
LANGFUSE_PUBLIC_KEY=<langfuse-public-key>
LANGFUSE_SECRET_KEY=<langfuse-secret-key>
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

배포 완료 후 확인

```text
https://<api-domain>/health
```

## 3. Streamlit Community Cloud 배포

- 저장소: `d-jskim/manufacturing-doc-rag`
- Branch: `main`
- Entrypoint: `frontend/app.py`
- Python: `3.12`

Advanced settings의 Secrets

```toml
RAG_API_URL = "https://<api-domain>"
CHAT_DB_PATH = "/tmp/manufacturing-doc-rag/chat.sqlite3"
LANGFUSE_PUBLIC_KEY = "<langfuse-public-key>"
LANGFUSE_SECRET_KEY = "<langfuse-secret-key>"
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
```

`frontend/requirements.txt`가 Streamlit 전용 의존성으로 사용됨

## 4. 배포 확인

- `/health`의 `status: ok`
- 샘플 사용자 로그인
- Basic 질문과 R1·R2·R3 요청
- 답변 문서·페이지·검색 문맥 표시
- Langfuse 서비스 trace 생성
- Streamlit 재시작 시 SQLite 대화가 초기화될 수 있다는 제한 확인

## 배포 제한

- Streamlit Community Cloud의 SQLite는 영구 저장소가 아님
- 장기 운영 시 PostgreSQL 등 외부 DB 필요
- R3 Cross-encoder는 CPU 환경에서 응답 지연 가능
- 실제 인증과 문서별 접근 권한은 미구현 상태
