"""검색 평가: 문서 단위 Hit@k·Recall@k와 보조 지표인 근거 문장(quote) 포함률."""

import re


def normalize(text: str) -> str:
    """PDF 추출 시 줄바꿈 위치가 원문과 다를 수 있으므로 공백·줄바꿈을 모두 지우고 비교한다."""
    return re.sub(r"\s+", "", text)


def reference_book_ids(item: dict) -> list[str]:
    return sorted({e["book_id"] for group in item["evidence_groups"] for e in group})


def check_quotes(items: list[dict], page_texts: dict[tuple[str, int], str]) -> list[dict]:
    """모든 quote가 해당 페이지 본문에 있는지 확인하고, 찾지 못한 근거 목록을 반환한다."""
    missing = []
    for item in items:
        for group in item["evidence_groups"]:
            for e in group:
                page_text = normalize(page_texts.get((e["book_id"], e["page"]), ""))
                if normalize(e["quote"]) not in page_text:
                    missing.append({"id": item["id"], "book_id": e["book_id"],
                                    "page": e["page"], "quote": e["quote"]})
    return missing


def score_question(item: dict, context: list[dict]) -> dict:
    references = reference_book_ids(item)
    row = {"id": item["id"], "k": len(context), "context_chunk_ids": [c["chunk_id"] for c in context]}
    if not references:
        return {**row, "applicable": False, "reason": "참고 근거 문서 없음"}

    found = sorted({c["book_id"] for c in context} & set(references))
    chunk_texts = [normalize(c["text"]) for c in context]
    groups = item["evidence_groups"]
    found_groups = sum(
        any(normalize(e["quote"]) in text for e in group for text in chunk_texts) for group in groups)
    return {
        **row,
        "applicable": True,
        "reference_book_ids": references,
        "found_book_ids": found,
        "hit": int(bool(found)),
        "recall": round(len(found) / len(references), 4),
        "quote_recall": round(found_groups / len(groups), 4),
    }


def summarize_scores(rows: list[dict]) -> dict:
    """해당 없음은 평균에서 빼고, 평가 대상 수와 해당 없음 수를 함께 남긴다."""
    scored = [r for r in rows if r["applicable"]]

    def mean(key: str) -> float | None:
        return round(sum(r[key] for r in scored) / len(scored), 4) if scored else None

    return {
        "evaluated_count": len(scored),
        "not_applicable_count": len(rows) - len(scored),
        "hit_at_k": mean("hit"),
        "recall_at_k": mean("recall"),
        "quote_recall_at_k": mean("quote_recall"),
        "mean_k": mean("k"),
    }