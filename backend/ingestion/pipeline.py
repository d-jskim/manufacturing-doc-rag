"""선택한 데이터셋·청킹·임베딩 설정으로 서비스 검색 인덱스를 준비한다."""

import hashlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml
from qdrant_client import QdrantClient

from backend.core.token_counter import HuggingFaceCounter
from backend.ingestion.chunk import chunk_pages, save_chunks
from backend.ingestion.extract import extract_folder
from backend.retrieval.vector_store import build_index

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def collection_name(dataset: str, chunk_version: str, embedding: str) -> str:
    return f"{dataset}_{chunk_version}_{embedding}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_summary(values: list[int]) -> dict:
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {"min": ordered[0], "median": median, "max": ordered[-1]}


def prepare_index(
    dataset: str,
    chunk_version: str,
    embedding: str,
    qdrant_url: str,
) -> dict:
    from backend.retrieval.embedder import Embedder

    documents_dir = ROOT / "data" / "raw" / dataset / "documents"
    if not documents_dir.exists():
        raise FileNotFoundError(f"문서 폴더가 없습니다: {documents_dir}")

    pdf_paths = sorted(documents_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"PDF 문서가 없습니다: {documents_dir}")

    chunk_config = load_yaml(
        ROOT / "configs" / "chunking" / f"{chunk_version}.yaml"
    )
    embedding_config = load_yaml(
        ROOT / "configs" / "embedding" / f"{embedding}.yaml"
    )
    if chunk_config["chunk_version"] != chunk_version:
        raise ValueError(f"청킹 설정 ID가 일치하지 않습니다: {chunk_version}")
    if embedding_config["model_code"] != embedding:
        raise ValueError(f"임베딩 설정 ID가 일치하지 않습니다: {embedding}")

    pages, failures = extract_folder(documents_dir)
    if failures:
        raise RuntimeError(f"PDF 추출 실패: {failures}")

    counter = HuggingFaceCounter(
        chunk_config["split_tokenizer"],
        chunk_config["prefix"],
    )
    chunks = chunk_pages(pages, chunk_config, counter)
    encoded_tokens = [chunk.chunk_encoded_token_count for chunk in chunks]
    chunks_dir = ROOT / "data" / "processed" / dataset / chunk_version
    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": dataset,
        "chunk_version": chunk_version,
        "embedding": embedding,
        "document_count": len(pdf_paths),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "document_hashes": {path.name: file_sha256(path) for path in pdf_paths},
        "chunk_config": chunk_config,
        "chunk_encoded_token_count": token_summary(encoded_tokens),
        "over_limit_chunk_count": sum(
            value > embedding_config["max_tokens"] for value in encoded_tokens
        ),
    }
    save_chunks(chunks, manifest, chunks_dir)

    chunks_path = chunks_dir / "chunks.jsonl"
    collection = collection_name(dataset, chunk_version, embedding)
    index_record = {
        "collection": collection,
        "chunks_sha256": file_sha256(chunks_path),
        "chunk_count": len(chunks),
        "embedding": embedding_config,
    }
    embedder = Embedder(embedding_config)
    client = QdrantClient(url=qdrant_url)
    built = build_index(
        client,
        collection,
        [asdict(chunk) for chunk in chunks],
        embedder,
        chunks_dir / f"index_{embedding}.json",
        index_record,
    )
    return {
        "dataset": dataset,
        "chunk_version": chunk_version,
        "embedding": embedding,
        "collection": collection,
        "index_reused": not built,
        "document_count": len(pdf_paths),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "over_limit_chunk_count": manifest["over_limit_chunk_count"],
        "chunks_path": chunks_path.relative_to(ROOT).as_posix(),
        "embedding_device": embedder.device,
    }
