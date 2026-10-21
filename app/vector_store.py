from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL, FAISS_DIR


def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize the Hugging Face embedding model.

    The model converts text into dense numerical vectors
    that can be used for semantic similarity search.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    documents: list[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Build a FAISS vector store from LangChain Documents.
    """

    if not documents:
        raise ValueError(
            "Cannot build a vector store without documents."
        )

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model,
    )

    return vector_store


def save_vector_store(
    vector_store: FAISS,
    directory: str | Path = FAISS_DIR,
) -> None:
    """
    Persist the FAISS index and document mappings to disk.
    """

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(str(directory))


def load_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    directory: str | Path = FAISS_DIR,
) -> FAISS:
    """
    Load a previously saved FAISS vector store.
    """

    directory = Path(directory)

    index_file = directory / "index.faiss"
    metadata_file = directory / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"No complete FAISS index found in: {directory}"
        )

    vector_store = FAISS.load_local(
        folder_path=str(directory),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

    return vector_store


def search_vector_store(
    vector_store: FAISS,
    question: str,
    k: int = 4,
) -> list[Document]:
    """
    Retrieve the k most similar document chunks.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    return vector_store.similarity_search(
        query=question,
        k=k,
    )