"""Qdrant 색인과 검색. 실행 조건 기록이 같으면 기존 색인을 재사용한다."""

import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models


def point_id(chunk_id: str) -> str:
    """Qdrant는 정수나 UUID만 ID로 받으므로, chunk_id로 항상 같은 UUID를 만든다."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def index_is_current(client: QdrantClient, collection: str, record_path: Path, index_record: dict) -> bool:
    if not record_path.exists() or not client.collection_exists(collection):
        return False
    saved = json.loads(record_path.read_text(encoding="utf-8"))
    point_count = client.count(collection, exact=True).count
    return saved == index_record and point_count == index_record["chunk_count"]


def build_index(client: QdrantClient, collection: str, chunks: list[dict], embedder,
                record_path: Path, index_record: dict, batch_size: int = 256) -> bool:
    """색인을 새로 만들었으면 True, 기존 색인을 재사용했으면 False를 반환한다."""
    if index_is_current(client, collection, record_path, index_record):
        return False
    if client.collection_exists(collection):
        client.delete_collection(collection)

    vectors = embedder.embed_passages([chunk["text"] for chunk in chunks])
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    points = [models.PointStruct(id=point_id(c["chunk_id"]), vector=v, payload=c)
              for c, v in zip(chunks, vectors)]
    for start in range(0, len(points), batch_size):
        client.upsert(collection, points=points[start:start + batch_size], wait=True)

    record_path.write_text(json.dumps(index_record, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def search(client: QdrantClient, collection: str, query_vector: list[float],
           limit: int, exact: bool) -> list[dict]:
    """exact=True는 모든 벡터와 하나하나 비교, False는 근사 검색(ANN)."""
    result = client.query_points(
        collection, query=query_vector, limit=limit,
        search_params=models.SearchParams(exact=exact), with_payload=True,
    )
    return [{**point.payload, "score": point.score} for point in result.points]