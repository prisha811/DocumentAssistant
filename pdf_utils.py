import re
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.
    """

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)

    return text.strip()


def load_pdf(pdf_path):
    """
    Read PDF page by page.
    """

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text()

        text = clean_text(text)

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    document.close()

    return pages


def chunk_pages(pages):
    """
    Split pages into chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    chunks = []

    chunk_id = 0

    for page in pages:

        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:

            chunks.append(
                {
                    "id": chunk_id,
                    "page": page["page"],
                    "text": chunk,
                }
            )

            chunk_id += 1

    return chunks