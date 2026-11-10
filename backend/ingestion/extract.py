"""PDF 문서를 페이지 단위로 추출한다."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader



@ dataclass
class Page:
    book_id: str    # 파일명에서 .pdf를 뺀 값(예: MB04)
    file_name: str  # 원본 파일명 (예: MB04.pdf)
    page: int       # 첫 페이지를 1로 하는 페이지 번호
    text: str
    
    
def extract_pdf(pdf_path: Path) -> list[Page]:
    """PDF 한 개를 페이지별로 추출한다."""
    reader = PdfReader(pdf_path)
    return [
        Page(
            book_id=pdf_path.stem,
            file_name=pdf_path.name,
            page=page_number,
            text=pdf_page.extract_text() or "",
        )
        for page_number, pdf_page in enumerate(reader.pages, start=1)
    ]
    
    
def extract_folder(documents_dir: Path) -> tuple[list[Page], list[dict]]:
    """폴더의 모든 PDF를 추출하고, 실패한 파일은 사유와 함께 따로 반환한다."""
    pages: list[Page] = []
    failures: list[dict] = []
    for pdf_path in sorted(documents_dir.glob("*.pdf")):
        try:
            pages.extend(extract_pdf(pdf_path))
        except Exception as error:
            failures.append({"file_name": pdf_path.name, "error": str(error)})
    return pages, failures