"""키워드 검색(BM25). 한국어는 조사가 붙어 같은 단어도 모양이 달라지므로, 글을 자르는 방식을 선택할 수 있게 한다."""

import re

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

CODE_PATTERN = re.compile(r"[A-Za-z]+-\d+[A-Za-z0-9-]*")  # 모델명·코드(예: CV-20)는 한 단어로 유지
CONTENT_TAGS = ("NNG", "NNP", "NR", "SL", "SN", "SH", "XR", "VV", "VA")  # 명사·외국어·숫자·어근·동사·형용사


def whitespace_tokenize(text: str) -> list[str]:
    """띄어쓰기로만 자른다. '내보내기는'과 '내보내기'가 서로 다른 단어가 된다."""
    return text.lower().split()


class KiwiTokenizer:
    """형태소 분석으로 조사·어미를 떼고 내용 단어만 남긴다."""

    def __init__(self) -> None:
        self._kiwi = Kiwi()

    def __call__(self, text: str) -> list[str]:
        codes = [code.lower() for code in CODE_PATTERN.findall(text)]
        morphs = [token.form.lower() for token in self._kiwi.tokenize(text)
                  if token.tag.split("-")[0] in CONTENT_TAGS]  # VV-R 같은 세부 태그도 앞부분으로 판단
        return codes + morphs


def make_tokenizer(name: str):
    if name == "whitespace":
        return whitespace_tokenize
    if name == "kiwi":
        return KiwiTokenizer()
    raise ValueError(f"알 수 없는 토크나이저: {name}")


class BM25Index:
    """청크 목록으로 BM25 색인을 메모리에 만들고, 질문과 단어가 겹치는 정도로 청크를 찾는다."""

    def __init__(self, chunks: list[dict], tokenizer):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self._bm25 = BM25Okapi([tokenizer(chunk["text"]) for chunk in chunks])

    def search(self, query: str, limit: int) -> list[dict]:
        scores = self._bm25.get_scores(self.tokenizer(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]