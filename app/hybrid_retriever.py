from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    DENSE_RETRIEVER_WEIGHT,
    BM25_RETRIEVER_WEIGHT,
    RETRIEVAL_CANDIDATE_K,
)

from pdf_utils import load_and_chunk_pdf

from vector_store import (
    create_embedding_model,
    build_vector_store,
    save_vector_store,
    load_vector_store,
)


def build_dense_retriever(
    documents: list[Document],
    embedding_model: HuggingFaceEmbeddings,
):
    """
    Build a dense retriever using a FAISS vector store.
    """

    vector_store = build_vector_store(
        documents=documents,
        embedding_model=embedding_model,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": RETRIEVAL_CANDIDATE_K,
        },
    )

    return retriever, vector_store


def build_bm25_retriever(
    documents: list[Document],
):
    """
    Build a sparse keyword retriever using BM25.
    """

    if not documents:
        raise ValueError(
            "Cannot build BM25 retriever without documents."
        )

    retriever = BM25Retriever.from_documents(
        documents=documents,
    )

    retriever.k = RETRIEVAL_CANDIDATE_K

    return retriever


def build_hybrid_retriever(
    documents: list[Document],
    embedding_model: HuggingFaceEmbeddings,
):
    """
    Combine dense FAISS retrieval and BM25 retrieval.

    Both retrievers return LangChain Document objects,
    preserving their source and page metadata.
    """

    dense_retriever, vector_store = build_dense_retriever(
        documents=documents,
        embedding_model=embedding_model,
    )

    bm25_retriever = build_bm25_retriever(
        documents=documents,
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            dense_retriever,
        ],
        weights=[
            BM25_RETRIEVER_WEIGHT,
            DENSE_RETRIEVER_WEIGHT,
        ],
    )

    return hybrid_retriever, vector_store


def main():
    """
    Run a simple hybrid retrieval demonstration.
    """

    pdf_files = sorted(
        Path("data/raw_pdfs").glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF found in data/raw_pdfs."
        )

    pdf_path = pdf_files[0]

    print(f"PDF: {pdf_path.name}")

    documents = load_and_chunk_pdf(pdf_path)

    print(f"Chunks loaded: {len(documents)}")

    print("\nLoading embedding model...")

    embedding_model = create_embedding_model()

    print("Embedding model loaded.")

    print("\nBuilding hybrid retriever...")

    hybrid_retriever, vector_store = build_hybrid_retriever(
        documents=documents,
        embedding_model=embedding_model,
    )

    print("Hybrid retriever created.")

    # Save the dense vector store for later use.
    save_vector_store(vector_store)

    print("FAISS vector store saved.")

    questions = [
        "What is retrieval augmented generation?",
        "Explain BM25 and keyword-based retrieval.",
        "How does FAISS perform similarity search?",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        results = hybrid_retriever.invoke(question)

        for rank, document in enumerate(results[:5], start=1):
            source = document.metadata.get(
                "source",
                "Unknown",
            )

            page = document.metadata.get(
                "page",
                "Unknown",
            )

            print(f"\nResult {rank}")
            print(f"Source: {source}")
            print(f"Page: {page}")
            print(f"Text: {document.page_content[:300]}")


if __name__ == "__main__":
    main()