from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks):

        self.chunks = chunks

        tokenized = [
            chunk["text"].lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized)

    def search(self, question, top_k=5):

        tokens = question.lower().split()

        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for idx, score in ranked[:top_k]:

            results.append({
                "text": self.chunks[idx]["text"],
                "page": self.chunks[idx]["page"],
                "score": float(score)
            })

        return results