"""데이터셋의 청크와 Qdrant 검색 인덱스를 준비한다."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ingestion.pipeline import prepare_index

DATASETS = tuple(
    sorted(
        path.name
        for path in (ROOT / "data" / "raw").iterdir()
        if (path / "documents").is_dir()
    )
)


def load_active_config() -> dict:
    path = ROOT / "configs" / "active.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    active = load_active_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--chunk-version", default=active["chunk_version"])
    parser.add_argument("--embedding", default=active["embedding"])
    parser.add_argument("--qdrant-url", default=active["qdrant_url"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_index(
        dataset=args.dataset,
        chunk_version=args.chunk_version,
        embedding=args.embedding,
        qdrant_url=args.qdrant_url,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
