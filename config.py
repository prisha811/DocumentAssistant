from pathlib import Path

# ==========================
# Project Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

RAW_PDF_DIR = DATA_DIR / "raw_pdfs"

FAISS_DIR = DATA_DIR / "faiss_index"

# ==========================
# Models
# ==========================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

LLM_MODEL = "gemma3:1b"

# ==========================
# Chunking
# ==========================

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100

# ==========================
# Retrieval
# ==========================

TOP_K = 10

SIMILARITY_THRESHOLD = 0.30