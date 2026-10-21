import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
FAISS_DIR = DATA_DIR / "faiss_index"

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:1b")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 10
SIMILARITY_THRESHOLD = 0.30
RETRIEVAL_CANDIDATE_K = 20
FINAL_CONTEXT_K = 2

# Hybrid retrieval weights
DENSE_RETRIEVER_WEIGHT = 0.5
BM25_RETRIEVER_WEIGHT = 0.5

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "")