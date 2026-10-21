from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class DocumentReranker:
    """
    Rerank retrieved document chunks using a cross-encoder.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        documents: list[Document],
        top_k: int = 4,
    ) -> list[Document]:
        """
        Score query-document pairs and return the top-ranked chunks.

        The original Document objects are returned, preserving
        their text and source metadata.
        """

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not documents:
            return []

        # Each pair contains the question and one candidate passage.
        pairs = [
            [question, document.page_content]
            for document in documents
        ]

        # The cross-encoder calculates a relevance score per pair.
        scores = self.model.predict(pairs)

        # Sort candidate indices by descending relevance score.
        ranked_indices = sorted(
            range(len(documents)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        # Return the original Documents in the new order.
        # This preserves source, page, and chunk metadata.
        reranked_documents = [
            documents[index]
            for index in ranked_indices[:top_k]
        ]

        return reranked_documents