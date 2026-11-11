"""RRF(Reciprocal Rank Fusion): 의미 검색과 키워드 검색의 결과 목록을 순위만으로 합친다.
두 검색의 점수(코사인 유사도, BM25)는 척도가 달라 직접 더할 수 없으므로, 각 목록의 순위를 1/(k+순위)로 바꿔 더한다."""


def rrf_fuse(dense: list[dict], keyword: list[dict], k: int, limit: int) -> list[dict]:
    """두 목록을 합쳐 RRF 점수가 높은 순으로 limit개를 반환한다.
    score에는 RRF 점수, dense_rank·keyword_rank에는 각 목록의 순위(1부터, 목록에 없으면 None)를 남긴다.
    점수가 같으면 먼저 들어온 순서(의미 검색 결과 → 키워드 검색 결과)를 유지한다."""
    fused: dict[str, dict] = {}
    for rank_key, results in (("dense_rank", dense), ("keyword_rank", keyword)):
        for rank, chunk in enumerate(results, start=1):
            entry = fused.setdefault(
                chunk["chunk_id"], {**chunk, "score": 0.0, "dense_rank": None, "keyword_rank": None})
            entry[rank_key] = rank
            entry["score"] += 1 / (k + rank)
    return sorted(fused.values(), key=lambda chunk: chunk["score"], reverse=True)[:limit]