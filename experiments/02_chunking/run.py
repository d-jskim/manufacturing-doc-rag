"""청킹 실험: c1·c2·c3 청크를 만들고 색인한 뒤, 같은 검색·문맥 한도로 검색 품질과 비용을 비교한다."""

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import yaml
from huggingface_hub import try_to_load_from_cache
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # backend·evaluation 모듈을 불러오기 위해 저장소 루트를 경로에 추가

from backend.core.cost import CostRecorder
from backend.core.token_counter import HuggingFaceCounter, TiktokenCounter
from backend.ingestion.chunk import Chunk, chunk_pages, resolve_chunk_size, save_chunks
from backend.ingestion.extract import extract_folder
from backend.retrieval.context import fill_context
from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_store import build_index, search
from evaluation.retrieval import check_quotes, score_question, summarize_scores

LIBRARIES = ["pypdf", "transformers", "tokenizers", "langchain-text-splitters",
             "sentence-transformers", "torch", "qdrant-client"]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def summarize(values: list[int]) -> dict:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def tokenizer_revision(model_name: str) -> str | None:
    """내려받은 토크나이저의 커밋 해시(캐시 폴더 이름)."""
    path = try_to_load_from_cache(model_name, "tokenizer_config.json")
    return Path(path).parent.name if isinstance(path, str) else None


def build_manifest(run_id: str, chunk_config: dict, counter: HuggingFaceCounter,
                   document_hashes: dict) -> dict:
    is_recursive = chunk_config["method"] == "recursive"
    return {
        "run_id": run_id,
        "chunk_config": chunk_config,
        "overhead_token_count": counter.count_overhead(),
        "resolved_chunk_size": resolve_chunk_size(chunk_config, counter) if is_recursive else None,
        "split_tokenizer": {"name": counter.name, "revision": tokenizer_revision(counter.name)},
        "document_hashes": document_hashes,
        "library_versions": {name: version(name) for name in LIBRARIES},
    }


def chunk_stats(chunks: list[Chunk], max_tokens: int) -> dict:
    encoded = [c.chunk_encoded_token_count for c in chunks]
    return {
        "chunk_count": len(chunks),
        "chunk_text_token_count": summarize([c.chunk_text_token_count for c in chunks]),
        "chunk_encoded_token_count": summarize(encoded),
        "over_limit_chunk_count": sum(t > max_tokens for t in encoded),
        "cross_page_chunk_count": sum(c.page_end > c.page_start for c in chunks),
    }


def compare_ann(exact_ids: list[str], ann_ids: list[str]) -> dict:
    """exact 상위 후보를 ANN이 얼마나 똑같이 돌려줬는지."""
    overlap = len(set(exact_ids) & set(ann_ids))
    return {"same_ranking": exact_ids == ann_ids,
            "ann_recall": round(overlap / len(exact_ids), 4) if exact_ids else None}


def run_dataset(config: dict, dataset: str, timestamp: str, embedder: Embedder,
                embedding_config: dict, retrieval_config: dict, client: QdrantClient) -> None:
    if dataset not in config["context_token_budget"]:
        raise SystemExit(f"{dataset}: 01 토큰 한도 baseline이 없어 실행하지 않음")
    token_budget = config["context_token_budget"][dataset]
    documents_dir = ROOT / config["paths"]["documents"].format(dataset=dataset)
    evaluation_dir = ROOT / config["paths"]["evaluation"].format(dataset=dataset)
    results_dir = ROOT / config["paths"]["results"].format(dataset=dataset)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. 추출(청킹 버전들이 함께 씀)
    start = time.perf_counter()
    pages, failed_files = extract_folder(documents_dir)
    extract_seconds = round(time.perf_counter() - start, 3)
    if failed_files:
        raise RuntimeError(f"PDF 추출 실패: {failed_files}")
    document_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(documents_dir.glob("*.pdf"))}

    # 2. 평가 질문과 quote 일치 사전 점검
    questions = [item for file_name in config["evaluation_files"]
                 for item in json.loads((evaluation_dir / file_name).read_text(encoding="utf-8"))]
    missing = check_quotes(questions, {(p.book_id, p.page): p.text for p in pages})
    quote_count = sum(len(group) for q in questions for group in q["evidence_groups"])
    print(f"{dataset}: quote {quote_count - len(missing)}/{quote_count}개 일치")
    if missing:
        for row in missing[:5]:
            print("  못 찾음:", row)
        raise SystemExit("quote 일치 점검 실패: 비교 규칙을 먼저 확인하세요.")

    generation_counter = TiktokenCounter(config["generation_tokenizer"]["model"],
                                         config["generation_tokenizer"]["fallback"])
    query_counter = HuggingFaceCounter(embedding_config["model"], embedding_config["query_prefix"])
    retrieval_version = retrieval_config["retrieval_version"]
    candidate_count = retrieval_config["candidate_count"]

    for chunk_version in config["chunk_versions"]:
        run_id = f"e02_{chunk_version}_{embedder.model_code}_p0_{retrieval_version}_{timestamp}"
        cost = CostRecorder()
        cost.stage_seconds["extract_shared"] = extract_seconds  # 같은 데이터셋의 청킹 버전들이 함께 씀

        # 3. 청킹
        chunk_config = load_yaml(ROOT / "configs" / "chunking" / f"{chunk_version}.yaml")
        counter = HuggingFaceCounter(chunk_config["split_tokenizer"], chunk_config["prefix"])
        with cost.stage("chunk"):
            chunks = chunk_pages(pages, chunk_config, counter)
        stats = chunk_stats(chunks, chunk_config["max_encoded_tokens"])
        manifest = build_manifest(run_id, chunk_config, counter, document_hashes)
        manifest["chunk_stats"] = stats
        chunks_dir = ROOT / config["paths"]["chunks"].format(dataset=dataset, chunk_version=chunk_version)
        save_chunks(chunks, manifest, chunks_dir)

        # 4. 임베딩·색인(청크와 임베딩 설정이 같으면 재사용)
        collection = f"{dataset}_{chunk_version}_{embedder.model_code}"
        index_record = {
            "collection": collection,
            "chunks_sha256": hashlib.sha256((chunks_dir / "chunks.jsonl").read_bytes()).hexdigest(),
            "chunk_count": len(chunks),
            "embedding": embedding_config,
        }
        with cost.stage("embed_index"):
            built = build_index(client, collection, [asdict(c) for c in chunks], embedder,
                                chunks_dir / f"index_{embedder.model_code}.json", index_record)
        cost.add_tokens("embedding_passage", sum(c.chunk_encoded_token_count for c in chunks) if built else 0)

        # 5. 질문별 검색·문맥 채우기·채점(exact로 계산, ANN은 비교용)
        rows = []
        with cost.stage("retrieval_eval"):
            for item in questions:
                query_vector = embedder.embed_query(item["question"])
                cost.add_tokens("embedding_query", query_counter.count(item["question"]))

                t0 = time.perf_counter()
                exact_candidates = search(client, collection, query_vector, candidate_count, exact=True)
                cost.add_request_seconds("search_exact", time.perf_counter() - t0)

                context, used = fill_context(exact_candidates, generation_counter, token_budget)
                row = score_question(item, context)
                row["context_token_count"] = used

                if retrieval_config["compare_ann"]:
                    t1 = time.perf_counter()
                    ann_candidates = search(client, collection, query_vector, candidate_count, exact=False)
                    cost.add_request_seconds("search_ann", time.perf_counter() - t1)
                    row["ann"] = compare_ann([c["chunk_id"] for c in exact_candidates],
                                             [c["chunk_id"] for c in ann_candidates])
                rows.append(row)

        ann_rows = [r["ann"] for r in rows if "ann" in r]
        summary = {
            "run_id": run_id,
            "dataset": dataset,
            "chunk_version": chunk_version,
            "embedding": embedder.model_code,
            "embedding_device": embedder.device,
            "retrieval": retrieval_version,
            "collection": collection,
            "index_reused": not built,
            "context_token_budget": token_budget,
            "candidate_count": candidate_count,
            "chunk_stats": stats,
            "retrieval_metrics": summarize_scores(rows),
            "ann_comparison": {
                "question_count": len(ann_rows),
                "same_ranking_count": sum(r["same_ranking"] for r in ann_rows),
                "mean_ann_recall": round(statistics.mean(r["ann_recall"] for r in ann_rows), 4) if ann_rows else None,
            },
            "cost": cost.to_dict(),
        }
        result = {"summary": summary, "questions": rows}
        (results_dir / f"{run_id}_retrieval.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        metrics = summary["retrieval_metrics"]
        ann = summary["ann_comparison"]
        print(f"{chunk_version}: Hit@k {metrics['hit_at_k']}, Recall@k {metrics['recall_at_k']}, "
              f"quote {metrics['quote_recall_at_k']}, 평균 k {metrics['mean_k']} | "
              f"ANN 순위 동일 {ann['same_ranking_count']}/{ann['question_count']} | "
              f"색인 {'재사용' if not built else '새로 만듦'}")


def main() -> None:
    """설정의 datasets를 차례로 실행한다. --dataset을 주면 그 하나만 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="하나만 실행할 때 데이터셋 이름(생략하면 설정의 datasets 전체)")
    args = parser.parse_args()

    config = load_yaml(Path(__file__).parent / "config.yaml")
    datasets = [args.dataset] if args.dataset else config["datasets"]
    embedding_config = load_yaml(ROOT / "configs" / "embedding" / f"{config['embedding']}.yaml")
    retrieval_config = load_yaml(ROOT / "configs" / "retrieval" / f"{config['retrieval']}.yaml")
    embedder = Embedder(embedding_config)
    client = QdrantClient(url=config["qdrant_url"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    for dataset in datasets:
        run_dataset(config, dataset, timestamp, embedder, embedding_config, retrieval_config, client)


if __name__ == "__main__":
    main()