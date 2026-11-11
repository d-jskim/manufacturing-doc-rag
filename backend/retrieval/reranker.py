"""하이브리드 검색 후보를 Cross-encoder로 재순위한다."""

from sentence_transformers import CrossEncoder


class Reranker:
    """질문과 청크 본문을 함께 평가해 관련도가 높은 순서로 정렬한다."""

    def __init__(self, config: dict):
        self.model_name = config["reranker_model"]
        self.device = config["reranker_device"]
        self.max_length = config["reranker_max_length"]
        self.batch_size = config["reranker_batch_size"]

        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )

    def rerank(self, question: str, candidates: list[dict]) -> list[dict]:
        """RRF 후보를 재채점하고 리랭커 점수가 높은 순서로 반환한다."""
        if not candidates:
            return []

        pairs = [[question, candidate["text"]] for candidate in candidates]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scored = []
        for candidate, raw_score in zip(candidates, scores, strict=True):
            rerank_score = float(
                raw_score.item() if hasattr(raw_score, "item") else raw_score
            )
            scored.append(
                {
                    **candidate,
                    "rrf_score": float(candidate["score"]),
                    "score": rerank_score,
                    "rerank_score": rerank_score,
                }
            )

        ranked = sorted(
            scored,
            key=lambda candidate: candidate["rerank_score"],
            reverse=True,
        )
        return [
            {**candidate, "rerank_rank": rank}
            for rank, candidate in enumerate(ranked, start=1)
        ]