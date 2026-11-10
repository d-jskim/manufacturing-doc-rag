"""페이지를 청크로 나눈다. c1은 페이지 단위, c2는 페이지 안에서 분할, c3는 페이지 경계를 넘어 분할한다."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.token_counter import HuggingFaceCounter
from backend.ingestion.extract import Page


@dataclass
class Chunk:
    chunk_id: str                   # {book_id}_{chunk_version}_{chunk_seq 4자리}
    book_id: str
    file_name: str
    chunk_version: str
    chunk_seq: int                  # 책 안에서 1부터 매기는 순번
    page_start: int
    page_end: int
    text: str
    char_count: int
    chunk_text_token_count: int     # 본문만
    chunk_encoded_token_count: int  # 접두어 + 본문 + 특수 토큰
    text_hash: str


def resolve_chunk_size(config: dict, counter: HuggingFaceCounter) -> int:
    """chunk_size가 auto이면 본문에 쓸 수 있는 최대 토큰 수를 계산한다."""
    if config["chunk_size"] == "auto":
        return config["max_encoded_tokens"] - counter.count_overhead()
    return int(config["chunk_size"])


def make_chunk(page: Page, version: str, seq: int, page_start: int, page_end: int,
               text: str, counter: HuggingFaceCounter) -> Chunk:
    return Chunk(
        chunk_id=f"{page.book_id}_{version}_{seq:04d}",
        book_id=page.book_id,
        file_name=page.file_name,
        chunk_version=version,
        chunk_seq=seq,
        page_start=page_start,
        page_end=page_end,
        text=text,
        char_count=len(text),
        chunk_text_token_count=counter.count_body(text),
        chunk_encoded_token_count=counter.count(text),
        text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def group_by_book(pages: list[Page]) -> dict[str, list[Page]]:
    books: dict[str, list[Page]] = {}
    for page in pages:
        books.setdefault(page.book_id, []).append(page)
    return books


def page_at(page_starts: list[tuple[int, int]], position: int) -> int:
    """책 전체 텍스트의 위치가 속한 페이지 번호."""
    page = page_starts[0][1]
    for start, number in page_starts:
        if start > position:
            break
        page = number
    return page


def split_across_pages(book_pages: list[Page], splitter) -> list[tuple[int, int, str]]:
    """c3: 페이지 표시([p.3])를 넣어 책 전체를 이은 뒤 나누고, 시작·끝 위치로 페이지 범위를 찾는다."""
    full_text = ""
    page_starts: list[tuple[int, int]] = []  # (페이지가 시작하는 위치, 페이지 번호)
    for page in book_pages:
        page_starts.append((len(full_text), page.page))
        full_text += f"[p.{page.page}]\n{page.text}\n\n"

    pieces = []
    search_from = 0
    for text in splitter.split_text(full_text):
        start = full_text.find(text, search_from)  # 앞 청크 다음 위치부터 찾음(overlap 단위와 무관)
        if start == -1:
            raise ValueError(f"청크 위치를 찾지 못함: {text[:30]}")
        end = start + len(text) - 1
        pieces.append((page_at(page_starts, start), page_at(page_starts, end), text))
        search_from = start + 1
    return pieces


def chunk_book(book_pages: list[Page], config: dict, counter: HuggingFaceCounter,
               splitter) -> list[Chunk]:
    if config["method"] == "page":
        pieces = [(page.page, page.page, page.text) for page in book_pages]
    elif not config["cross_page"]:
        pieces = [(page.page, page.page, text)
                  for page in book_pages for text in splitter.split_text(page.text)]
    else:
        pieces = split_across_pages(book_pages, splitter)

    pieces = [piece for piece in pieces if piece[2].strip()]  # 빈 페이지 제외
    first = book_pages[0]
    return [make_chunk(first, config["chunk_version"], seq, start, end, text, counter)
            for seq, (start, end, text) in enumerate(pieces, start=1)]


def chunk_pages(pages: list[Page], config: dict, counter: HuggingFaceCounter) -> list[Chunk]:
    """설정(configs/chunking/c*.yaml)에 따라 전체 페이지를 청크로 나눈다."""
    splitter = None
    if config["method"] == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=resolve_chunk_size(config, counter),
            chunk_overlap=config["chunk_overlap"],
            length_function=counter.count_body,
        )

    chunks: list[Chunk] = []
    for book_pages in group_by_book(pages).values():
        chunks += chunk_book(book_pages, config, counter, splitter)

    # 분할한 청크는 실제 인코딩 길이로 한도 이하인지 검증한다(c1은 초과가 정상이므로 제외)
    if config["method"] == "recursive":
        over = [c.chunk_id for c in chunks if c.chunk_encoded_token_count > config["max_encoded_tokens"]]
        if over:
            raise ValueError(f"인코딩 길이 한도 초과 청크 {len(over)}개: {over[:5]}")
    return chunks


def save_chunks(chunks: list[Chunk], manifest: dict, out_dir: Path) -> None:
    """chunks.jsonl(청크 한 개 = 한 줄)과 manifest.json을 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")