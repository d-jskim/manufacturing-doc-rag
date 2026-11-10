"""청킹 실험 1단계: c1·c2·c3 청크를 만들어 저장하고, 버전별 청크 길이 통계를 결과로 남긴다."""

import hashlib
import json
import statistics
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import yaml
from huggingface_hub import try_to_load_from_cache

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # backend 모듈을 불러오기 위해 저장소 루트를 경로에 추가

from backend.core.token_counter import HuggingFaceCounter
from backend.ingestion.chunk import Chunk, chunk_pages, resolve_chunk_size, save_chunks
from backend.ingestion.extract import extract_folder

LIBRARIES = ["pypdf", "transformers", "tokenizers", "langchain-text-splitters"]


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


def main() -> None:
    config = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text(encoding="utf-8"))
    dataset = config["dataset"]
    documents_dir = ROOT / config["paths"]["documents"].format(dataset=dataset)
    results_dir = ROOT / config["paths"]["results"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    pages, failed_files = extract_folder(documents_dir)
    if failed_files:
        raise RuntimeError(f"PDF 추출 실패: {failed_files}")
    document_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(documents_dir.glob("*.pdf"))}

    results_dir.mkdir(parents=True, exist_ok=True)
    for chunk_version in config["chunk_versions"]:
        run_id = f"e02_{chunk_version}_p0_r0_{timestamp}"
        chunk_config = yaml.safe_load(
            (ROOT / "configs" / "chunking" / f"{chunk_version}.yaml").read_text(encoding="utf-8"))
        counter = HuggingFaceCounter(chunk_config["split_tokenizer"], chunk_config["prefix"])
        max_tokens = chunk_config["max_encoded_tokens"]

        chunks = chunk_pages(pages, chunk_config, counter)
        stats = chunk_stats(chunks, max_tokens)

        manifest = build_manifest(run_id, chunk_config, counter, document_hashes)
        manifest["chunk_stats"] = stats
        out_dir = ROOT / config["paths"]["chunks"].format(dataset=dataset, chunk_version=chunk_version)
        save_chunks(chunks, manifest, out_dir)

        result = {"summary": {"run_id": run_id, "dataset": dataset, **stats}}
        (results_dir / f"{run_id}_chunk_stats.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        note = " (c1은 페이지 단위라 초과가 정상)" if chunk_config["method"] == "page" else ""
        print(f"{chunk_version}: 청크 {stats['chunk_count']}개, "
              f"{max_tokens} 초과 {stats['over_limit_chunk_count']}개, "
              f"페이지를 넘는 청크 {stats['cross_page_chunk_count']}개{note}")


if __name__ == "__main__":
    main()