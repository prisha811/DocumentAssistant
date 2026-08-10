from sentence_transformers import CrossEncoder
from collections import deque
import re
import ollama

from config import LLM_MODEL

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from bm25 import BM25Retriever

def extract_definition_term(question):
    question = question.lower().strip()

    patterns = [
        r"what is an (.+?)\?",
        r"what is a (.+?)\?",
        r"what are (.+?)\?",
        r"define (.+?)$",
        r"explain (.+?)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group(1).strip()

    return ""


def is_definition_question(question):
    question = question.lower().strip()

    definition_patterns = [
        r"^what is (an |a )?.+\?$",
        r"^what are .+\?$",
        r"^define .+$",
        r"^explain .+$",
    ]

    return any(
        re.match(pattern, question)
        for pattern in definition_patterns
    )


def definition_relevance(question, text):
    question = question.lower()
    text_lower = text.lower()

    term = extract_definition_term(question)

    if not term:
        return 0

    score = 0

    # Exact term match
    if term in text_lower:
        score += 3

    # Expanded terms
    expansions = {
        "llm": "large language model",
        "llms": "large language models",
        "nlp": "natural language processing",
    }

    expansion = expansions.get(term)

    if expansion and expansion in text_lower:
        score += 5

    # Prefer passages that actually define the concept
    definition_phrases = [
        "is a",
        "is an",
        "refers to",
        "defined as",
        "known as",
        "is the",
        "are a",
        "are an",
    ]

    for phrase in definition_phrases:
        if phrase in text_lower:
            score += 1

    # Penalize passages that are talking about the term in another context
    unrelated_contexts = [
        "llm-as-a-judge",
        "llm as a judge",
        "evaluation",
        "benchmark",
        "scoring",
        "ranking",
        "judge",
    ]

    if term in ["llm", "llms", "large language model", "large language models"]:
        for phrase in unrelated_contexts:
            if phrase in text_lower:
                score -= 2

    return score

def reciprocal_rank_fusion(*rankings, k=60):

    scores = {}

    for ranking in rankings:

        for rank, chunk in enumerate(ranking):

            key = chunk["text"]

            if key not in scores:

                scores[key] = {
                    "chunk": chunk,
                    "score": 0
                }

            scores[key]["score"] += 1 / (k + rank + 1)

    results = sorted(
        scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return [
        item["chunk"]
        for item in results
    ]


import re


def get_question_type(question):
    q = question.lower().strip()

    if re.match(r"^(what is|what are|define|explain|what does|what do)\b", q):
        return "definition"

    if any(word in q for word in [
        "compare",
        "difference",
        "differentiate",
        "contrast",
        "versus",
        " vs ",
    ]):
        return "comparison"

    if any(word in q for word in [
        "how does",
        "how do",
        "how is",
        "how are",
        "how can",
        "why does",
        "why do",
        "why is",
        "why are",
        "why can",
    ]):
        return "reasoning"

    if any(word in q for word in [
        "process",
        "steps",
        "workflow",
        "pipeline",
        "procedure",
    ]):
        return "process"

    return "general"


def extract_main_terms(question):
    q = question.lower()

    # Important abbreviations in the document
    aliases = {
        "llm": "large language model",
        "llms": "large language models",
        "nlp": "natural language processing",
        "rag": "retrieval augmented generation",
        "bm25": "bm25",
        "ann": "approximate nearest neighbor",
    }

    terms = []

    for short, expanded in aliases.items():
        if re.search(r"\b" + re.escape(short) + r"\b", q):
            terms.append(short)
            terms.append(expanded)

    return terms


def question_aware_score(question, chunk):
    question_type = get_question_type(question)

    text = chunk["text"].lower()
    terms = extract_main_terms(question)

    score = 0.0

    # ---------------------------------
    # Exact concept matching
    # ---------------------------------
    for term in terms:
        if term in text:
            score += 2.0

    # ---------------------------------
    # Definition questions
    # ---------------------------------
    if question_type == "definition":

        definition_patterns = [
            " is a ",
            " is an ",
            " is the ",
            " refers to ",
            " defined as ",
            " known as ",
            " can be defined as ",
            " are models ",
            " are systems ",
        ]

        for pattern in definition_patterns:
            if pattern in text:
                score += 2.0

        # Penalize passages that merely mention the concept
        # as a subtopic or application.
        mention_patterns = [
            "task in",
            "tasks in",
            "application of",
            "applications of",
            "used for",
            "used to",
            "based agents",
            "based systems",
        ]

        for pattern in mention_patterns:
            if pattern in text:
                score -= 1.5

    # ---------------------------------
    # Comparison questions
    # ---------------------------------
    elif question_type == "comparison":

        # Reward chunks containing multiple concepts.
        if len(terms) >= 2:
            matched_terms = sum(
                1 for term in terms
                if term in text
            )

            if matched_terms >= 2:
                score += 4.0

    # ---------------------------------
    # Reasoning questions
    # ---------------------------------
    elif question_type == "reasoning":

        reasoning_patterns = [
            "because",
            "due to",
            "therefore",
            "however",
            "while",
            "although",
            "enables",
            "allows",
            "mitigates",
            "improves",
            "reduces",
            "increases",
        ]

        for pattern in reasoning_patterns:
            if pattern in text:
                score += 0.5

    # ---------------------------------
    # Process questions
    # ---------------------------------
    elif question_type == "process":

        process_patterns = [
            "first",
            "then",
            "next",
            "finally",
            "step",
            "process",
            "pipeline",
            "workflow",
            "retrieve",
            "generate",
        ]

        for pattern in process_patterns:
            if pattern in text:
                score += 0.5

    return score

class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

def rerank(
    self,
    question,
    chunks,
    top_k=5
):
    if not chunks:
        return []

    pairs = [
        (
            question,
            chunk["text"]
        )
        for chunk in chunks
    ]

    # Cross-Encoder relevance score
    cross_scores = self.model.predict(pairs)

    ranked = []

    for cross_score, chunk in zip(cross_scores, chunks):

        rule_score = question_aware_score(
            question,
            chunk
        )

        final_score = (
            float(cross_score)
            + rule_score
        )

        ranked.append(
            (
                final_score,
                chunk
            )
        )

    ranked.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        chunk
        for _, chunk in ranked[:top_k]
    ]


class ConversationMemory:

    def __init__(self, max_history=6):

        self.history = deque(maxlen=max_history)

    def add(self, role, message):

        self.history.append({
            "role": role,
            "message": message
        })

    def get_context(self):

        if not self.history:
            return ""

        context = ""

        for item in self.history:

            context += (
                f"{item['role']}: "
                f"{item['message']}\n"
            )

        return context

    def clear(self):

        self.history.clear()




def generate_answer(question, retrieved_chunks, conversation=""):

    context = ""

    for chunk in retrieved_chunks:
        context += f"[Page {chunk['page']}]\\n{chunk['text']}\\n\\n"

    definition_question = is_definition_question(question)
    definition_instruction = ""

    if definition_question:
        definition_instruction = """
This is a definition question.

- Identify the exact term being defined.
- Prefer a passage that explicitly defines or introduces that term.
- Do NOT answer using a passage that merely mentions the term in an example,
  benchmark, evaluation method, application, or related concept.
- Do NOT turn an LLM evaluation/use case into a definition of the LLM itself.
- If the document does not explicitly support a definition, say that the
  information could not be found instead of using outside knowledge.
"""

    system_prompt = f"""
You are DocumentAssistant.

Answer ONLY from the provided document context.

Rules:

- Use only information explicitly supported by the document.
- Do not use outside knowledge.
- Answer the exact question asked.
- Prefer the most directly relevant passage over passages that merely contain
  the same keyword.
- Do not combine unrelated passages just because they contain the same term.
- Explain naturally and concisely.
- Do not repeat the context.
- Do not output the prompt.
- Do not mention "Context" or "Previous Conversation".
- If the answer is not supported by the provided context, reply exactly:

I couldn't find this information in the uploaded document.
{definition_instruction}
"""

    user_prompt = f"""
Question:
{question}

Document:

{context}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0.1,
            "num_predict": 350,
        }
    )

    return response["message"]["content"].strip()


from config import (
    EMBEDDING_MODEL,
    FAISS_DIR,
    TOP_K,
    SIMILARITY_THRESHOLD
)

# -------------------------
# Load embedding model
# -------------------------

model = SentenceTransformer(EMBEDDING_MODEL)

# -------------------------
# Cache paths
# -------------------------

INDEX_FILE = FAISS_DIR / "index.faiss"
CHUNK_FILE = FAISS_DIR / "chunks.pkl"

# -------------------------
# Generate embeddings
# -------------------------

def embed_texts(texts):

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.astype("float32")


# -------------------------
# Build FAISS Index
# -------------------------

def build_index(chunks):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embed_texts(texts)

    dimension = embeddings.shape[1]

    # Cosine Similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# -------------------------
# Save Index
# -------------------------

def save_index(index, chunks):

    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    with open(CHUNK_FILE, "wb") as f:
        pickle.dump(chunks, f)


# -------------------------
# Load Index
# -------------------------

def load_index():

    index = faiss.read_index(str(INDEX_FILE))

    with open(CHUNK_FILE, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


# -------------------------
# Cache Check
# -------------------------

def cache_exists():

    return INDEX_FILE.exists() and CHUNK_FILE.exists()


# -------------------------
# Retrieve
# -------------------------

def retrieve(question, index, chunks):
    query_embedding = embed_texts([question])

    # Retrieve a larger candidate pool.
    candidate_k = max(TOP_K * 4, 20)

    scores, indices = index.search(
        query_embedding,
        candidate_k
    )

    retrieved = []
    seen = set()

    for score, idx in zip(scores[0], indices[0]):

        # Ignore invalid index
        if idx == -1:
            continue

        # Similarity filter
        if score < SIMILARITY_THRESHOLD:
            continue

        chunk = chunks[idx]

        # Remove duplicate chunks
        if chunk["text"] in seen:
            continue

        seen.add(chunk["text"])

        retrieved.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "score": round(float(score), 3)
        })

    return retrieved



class DocumentAssistant:

    def __init__(self, index, chunks):

        self.index = index
        self.chunks = chunks

        # Initialize retrievers
        self.bm25 = BM25Retriever(chunks)
        self.reranker = Reranker()

        # Conversation memory
        self.memory = ConversationMemory()

    def ask(self, question):

        question = question.strip()
        lower = question.lower()

        # -------------------------
        # Greetings
        # -------------------------

        greetings = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]

        if lower in greetings:
            return (
                "Hello! 👋\nI'm DocumentAssistant.\nAsk me anything about the uploaded document.",
                [],
            )

        # -------------------------
        # Thanks
        # -------------------------

        thanks = [
            "thanks",
            "thank you",
            "ty",
        ]

        if lower in thanks:
            return ("You're welcome! 😊", [])

        # -------------------------
        # Goodbye
        # -------------------------

        goodbye = [
            "bye",
            "goodbye",
            "see you",
        ]

        if lower in goodbye:
            return ("Goodbye! 👋", [])

        # -------------------------
        # FAISS Retrieval
        # -------------------------

        faiss_results = retrieve(
            question,
            self.index,
            self.chunks,
        )

        # -------------------------
        # BM25 Retrieval
        # -------------------------

        bm25_query = question

        if is_definition_question(question):
            term = extract_definition_term(question)
            expansions = {
                "llm": "large language model",
                "llms": "large language models",
                "nlp": "natural language processing",
            }
            expansion = expansions.get(term.lower(), "") if term else ""

            if expansion:
                bm25_query = f"{term} {expansion} definition"

        bm25_results = self.bm25.search(
            bm25_query,
            top_k=max(TOP_K * 4, 20)
        )

        # -------------------------
        # Reciprocal Rank Fusion
        # -------------------------

        retrieved_chunks = reciprocal_rank_fusion(
            faiss_results,
            bm25_results,
        )

        if is_definition_question(question):
            retrieved_chunks.sort(
                key=lambda chunk: definition_relevance(
                    question,
                    chunk["text"]
                ),
                reverse=True
            )

        # -------------------------
        # Cross Encoder Reranking
        # -------------------------

        retrieved_chunks = self.reranker.rerank(
            question,
            retrieved_chunks,
        )

        # -------------------------
        # Remove Duplicates
        # -------------------------

        unique = {}

        for chunk in retrieved_chunks:
            unique[chunk["text"]] = chunk

        retrieved_chunks = list(unique.values())

        # -------------------------
        # Keep Best 5
        # -------------------------

        retrieved_chunks = retrieved_chunks[:4]

        for chunk in retrieved_chunks:
            chunk.pop("_definition_score", None)

        # -------------------------
        # No Result Found
        # -------------------------

        if len(retrieved_chunks) == 0:
            return (
                "I couldn't find anything related to your question in the uploaded document.",
                [],
            )

        # -------------------------
        # Generate Answer
        # -------------------------

        answer = generate_answer(
            question,
            retrieved_chunks,
        )

        # -------------------------
        # Save Conversation
        # -------------------------

        self.memory.add("User", question)
        self.memory.add("Assistant", answer)

        return answer, retrieved_chunks