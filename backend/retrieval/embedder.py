"""설정(configs/embedding/*.yaml)에 따라 문서와 질문을 벡터로 바꾼다."""

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, config: dict):
        self.model_code = config["model_code"]
        self.model_name = config["model"]
        self.passage_prefix = config["passage_prefix"]
        self.query_prefix = config["query_prefix"]
        self.batch_size = config["batch_size"]
        self._model = SentenceTransformer(self.model_name)  # GPU가 있으면 자동으로 GPU 사용
        self.device = str(self._model.device)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            [self.passage_prefix + text for text in texts],
            batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=True,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()