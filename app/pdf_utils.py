from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """
    Load a PDF into LangChain Document objects.

    Each page becomes a separate Document.
    Page metadata is retained for source citations.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The supplied file must be a PDF.")

    loader = PyMuPDFLoader(str(pdf_path))

    documents = loader.load()

    # Convert PyMuPDFLoader's zero-based page metadata
    # into human-readable, one-based page numbers.
    for document in documents:
        document.metadata["page"] = (
            document.metadata.get("page", 0) + 1
        )

        document.metadata["source"] = pdf_path.name

    # Remove empty pages.
    documents = [
        document
        for document in documents
        if document.page_content.strip()
    ]

    if not documents:
        raise ValueError(
            "The PDF contains no extractable text."
        )

    return documents


def chunk_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Split loaded documents into smaller chunks.

    Each chunk retains its source page metadata.
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
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # Assign a unique ID to every chunk.
    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id

    return chunks


def load_and_chunk_pdf(
    pdf_path: str | Path,
) -> list[Document]:
    """
    Load a PDF and split its pages into chunks.
    """

    documents = load_pdf(pdf_path)

    return chunk_documents(documents)