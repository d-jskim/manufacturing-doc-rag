"""로컬 Qdrant Collection을 배포용 Qdrant로 복사한다."""

import argparse
import os

from qdrant_client import QdrantClient, models

DEFAULT_COLLECTIONS = (
    "manufacturing_basic_c1_m1",
    "manufacturing_basic_c2_m1",
    "manufacturing_basic_c3_m1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-url",
        default=os.getenv("QDRANT_SOURCE_URL", "http://127.0.0.1:6333"),
    )
    parser.add_argument(
        "--source-api-key",
        default=os.getenv("QDRANT_SOURCE_API_KEY"),
    )
    parser.add_argument("--target-url", default=os.getenv("QDRANT_TARGET_URL"))
    parser.add_argument("--target-api-key", default=os.getenv("QDRANT_TARGET_API_KEY"))
    parser.add_argument("--collections", nargs="+", default=DEFAULT_COLLECTIONS)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def copy_collection(
    source: QdrantClient,
    target: QdrantClient,
    collection: str,
    recreate: bool,
) -> int:
    if target.collection_exists(collection):
        if not recreate:
            raise RuntimeError(
                f"대상 Collection이 이미 존재합니다: {collection}. "
                "다시 만들려면 --recreate를 사용하세요."
            )
        target.delete_collection(collection)

    source_info = source.get_collection(collection)
    target.create_collection(
        collection_name=collection,
        vectors_config=source_info.config.params.vectors,
        sparse_vectors_config=source_info.config.params.sparse_vectors,
    )

    copied = 0
    offset = None
    while True:
        points, offset = source.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if points:
            target.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(
                        id=point.id,
                        vector=point.vector,
                        payload=point.payload,
                    )
                    for point in points
                ],
                wait=True,
            )
        copied += len(points)
        if offset is None:
            return copied


def main() -> None:
    args = parse_args()
    if not args.target_url or not args.target_api_key:
        raise SystemExit("QDRANT_TARGET_URL과 QDRANT_TARGET_API_KEY가 필요합니다.")

    source = QdrantClient(
        url=args.source_url,
        api_key=args.source_api_key or None,
    )
    target = QdrantClient(url=args.target_url, api_key=args.target_api_key)
    for collection in args.collections:
        count = copy_collection(source, target, collection, args.recreate)
        print(f"{collection}: {count}개 복사 완료")


if __name__ == "__main__":
    main()
