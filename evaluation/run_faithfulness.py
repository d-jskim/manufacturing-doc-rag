"""세 검색 방식의 답변에 RAGAS Faithfulness를 적용한다."""

import json
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from backend.core.token_counter import TiktokenCounter
from backend.retrieval.context import fill_context

RESULTS = (
    ROOT
    / "experiments"
    / "04_retrieval"
    / "results"
    / "manufacturing_basic"
)
ANSWER_PATH = RESULTS / "e04_c1_m1_p1_all_20261111_1553.json"
RETRIEVAL_PATH = RESULTS / "e04_c1_m1_p1_k235_20261111_1415.json"
OUTPUT_PATH = RESULTS / "e04_c1_m1_p1_faithfulness_20261111_1553.json"
CHUNKS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "manufacturing_basic"
    / "c1"
    / "chunks.jsonl"
)
MODES = ("naive", "hybrid", "rerank")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for mode in MODES:
        scores = [
            row["faithfulness"]
            for row in rows
            if row["retrieval_mode"] == mode
            and row["faithfulness"] is not None
        ]
        summary[mode] = {
            "mean_faithfulness": (
                round(sum(scores) / len(scores), 4)
                if scores
                else None
            ),
            "evaluated_count": len(scores),
            "error_count": sum(
                row["retrieval_mode"] == mode
                and row["faithfulness"] is None
                for row in rows
            ),
        }
    return summary


def save(rows: list[dict]) -> None:
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "metric": "ragas_faithfulness",
                "model": "gpt-6-luna",
                "answer_result": str(ANSWER_PATH.relative_to(ROOT)),
                "retrieval_result": str(RETRIEVAL_PATH.relative_to(ROOT)),
                "summary": summarize(rows),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    answer_records = load_json(ANSWER_PATH)["records"]
    retrieval_result = load_json(RETRIEVAL_PATH)
    chunks = {
        chunk["chunk_id"]: chunk
        for chunk in (
            json.loads(line)
            for line in CHUNKS_PATH.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
    }
    context_ids = {
        (row["id"], mode): row["context_chunk_ids"]
        for mode in MODES
        for row in retrieval_result["rows"][mode]["k3"]
    }

    question_ids = list(
        dict.fromkeys(
            record["question_id"]
            for record in answer_records
        )
    )
    if limit is not None:
        selected_ids = set(question_ids[:limit])
        answer_records = [
            record
            for record in answer_records
            if record["question_id"] in selected_ids
        ]

    active = yaml.safe_load(
        (ROOT / "configs" / "active.yaml").read_text(
            encoding="utf-8"
        )
    )
    counter = TiktokenCounter("gpt-6-luna", "o200k_base")

    rows = (
        load_json(OUTPUT_PATH)["rows"]
        if OUTPUT_PATH.exists()
        else []
    )
    completed = {
        (row["question_id"], row["retrieval_mode"])
        for row in rows
    }

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        timeout=180.0,
    )
    llm = llm_factory(
        model="gpt-6-luna",
        provider="openai",
        client=client,
        seed=42,
        max_tokens=2048,
    )
    metric = Faithfulness(llm=llm)

    for record in answer_records:
        key = (
            record["question_id"],
            record["retrieval_mode"],
        )
        if key in completed:
            continue

        selected = [
            chunks[chunk_id]
            for chunk_id in context_ids[key]
        ]
        context, token_count = fill_context(
            selected,
            counter,
            active["context_token_budget"],
        )
        response = record["response"]
        answer = response["answer"]
        if response.get("reason"):
            answer += f"\n사유: {response['reason']}"

        started = time.perf_counter()
        try:
            result = metric.score(
                user_input=record["question"],
                response=answer,
                retrieved_contexts=[
                    chunk["text"]
                    for chunk in context
                ],
            )
            score = float(result.value)
            error = None
        except Exception as exc:
            score = None
            error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "question_id": record["question_id"],
                "retrieval_mode": record["retrieval_mode"],
                "status": response["status"],
                "context_chunk_ids": [
                    chunk["chunk_id"]
                    for chunk in context
                ],
                "context_token_count": token_count,
                "faithfulness": score,
                "seconds": round(
                    time.perf_counter() - started,
                    3,
                ),
                "error": error,
            }
        )
        save(rows)
        print(f"{key[0]} / {key[1]}: {score}")

    print(
        json.dumps(
            summarize(rows),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()