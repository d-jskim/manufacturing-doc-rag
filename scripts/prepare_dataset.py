"""데이터셋 ZIP을 data/raw/{dataset}/에 풀고, PDF 수를 기대값과 비교해 기록한다."""

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    """큰 ZIP도 메모리에 다 올리지 않도록 1MB씩 읽어 해시를 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="configs/datasets.yaml의 데이터셋 이름(예: manufacturing_basic)")
    parser.add_argument("--force", action="store_true", help="이미 준비된 폴더가 있어도 다시 풂")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text(encoding="utf-8"))[args.dataset]
    source_dir = (ROOT / config["source"]).resolve()
    target_dir = ROOT / "data" / "raw" / args.dataset
    documents_dir = target_dir / "documents"
    expected = config["expected_pdf_count"]

    existing = len(list(documents_dir.glob("*.pdf"))) if documents_dir.exists() else 0
    if existing == expected and not args.force:
        print(f"{args.dataset}: 이미 준비됨(PDF {existing}/{expected}개). 다시 풀려면 --force")
        return

    start = time.perf_counter()
    zip_paths = sorted(source_dir.glob("*.zip"))
    if not zip_paths:
        sys.exit(f"ZIP 파일 없음: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target_dir)  # ZIP 안에 documents/, evaluation/ 폴더가 들어 있음

    pdf_count = len(list(documents_dir.glob("*.pdf")))
    manifest = {
        "dataset": args.dataset,
        "source_dir": config["source"],
        "zip_files": [{"name": p.name, "sha256": sha256(p)} for p in zip_paths],
        "expected_pdf_count": expected,
        "pdf_count": pdf_count,
        "pdf_count_matches": pdf_count == expected,
        "evaluation_files": sorted(p.name for p in (target_dir / "evaluation").glob("*.json")),
        "elapsed_seconds": round(time.perf_counter() - start, 1),
    }
    (target_dir / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.dataset}: PDF {pdf_count}/{expected}개, ZIP {len(zip_paths)}개, {manifest['elapsed_seconds']}초")
    if pdf_count != expected:
        sys.exit("PDF 수가 기대값과 다릅니다. prepare_manifest.json을 확인하세요.")


if __name__ == "__main__":
    main()