📄 DocumentAssistant

A PDF question-answering application built using Retrieval-Augmented Generation (RAG).

DocumentAssistant lets a user upload a PDF and ask questions about its contents through a Streamlit chat interface. Instead of sending the entire document directly to an LLM, the application retrieves relevant passages first and uses those passages as context for answer generation.

✨ Features

PDF upload through Streamlit

PDF text extraction with PyMuPDF

Overlapping text chunking

Dense semantic retrieval with FAISS

Lexical retrieval with BM25

Hybrid retrieval using FAISS + BM25

Reciprocal Rank Fusion (RRF)

Cross-Encoder reranking

Local LLM inference using Ollama

Qwen 2.5 7B for answer generation

Source page references for retrieved content

Cached FAISS index

Conversation interface with chat history

🧠 RAG Pipeline

                         PDF
                          │
                          ▼
                 Text Extraction
                   (PyMuPDF)
                          │
                          ▼
                    Chunking
                          │
                          ▼
                 Text Embeddings
            all-MiniLM-L6-v2
                          │
                          ▼
                    FAISS Index
                          │
                          │
                    USER QUERY
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      Semantic Retrieval        Lexical Retrieval
           (FAISS)                   (BM25)
             │                         │
             └────────────┬────────────┘
                          ▼
                Reciprocal Rank
                  Fusion (RRF)
                          │
                          ▼
                Cross-Encoder
                  Reranking
                          │
                          ▼
                 Top Relevant
                     Chunks
                          │
                          ▼
                Context + Query
                          │
                          ▼
                 Qwen 2.5 7B
                  via Ollama
                          │
                          ▼
                 Answer + Sources

🔍 How It Works

1. PDF Processing

The uploaded PDF is read page by page using PyMuPDF. Page numbers are preserved as metadata so the application can display the source pages used for an answer.

2. Chunking

The extracted document text is divided into smaller overlapping chunks.

Current configuration:

Chunk size: 500
Chunk overlap: 100

Chunking makes the document suitable for embedding and retrieval.

3. Embeddings

Each text chunk is converted into a dense vector representation using:

sentence-transformers/all-MiniLM-L6-v2

The embeddings are normalized before being stored in FAISS.

4. FAISS Semantic Retrieval

FAISS performs dense vector similarity search over the document chunks.

The project uses:

FAISS IndexFlatIP

with normalized vectors.

5. BM25 Lexical Retrieval

The application also uses BM25 for keyword-based retrieval.

BM25 is useful when exact terminology matters, while dense retrieval is useful for semantic similarity.

FAISS → semantic similarity
BM25  → lexical / keyword matching

6. Reciprocal Rank Fusion

The FAISS and BM25 results are combined using Reciprocal Rank Fusion (RRF).

FAISS ranking ──┐
                ├──→ RRF → unified ranking
BM25 ranking ───┘

RRF combines the ranked results without requiring the raw scores of the two retrievers to be directly comparable.

7. Cross-Encoder Reranking

The retrieved candidates are passed through:

cross-encoder/ms-marco-MiniLM-L-6-v2

The Cross-Encoder evaluates the question and candidate passage together and reranks the candidates according to query-passage relevance.

8. Context Construction

The highest-ranked chunks are assembled into a context containing their page numbers and text.

The LLM receives:

User Question
+
Retrieved Document Context

rather than the entire PDF.

9. Local LLM Generation

The application uses:

Qwen 2.5 7B

through Ollama.

The LLM is instructed to answer using the retrieved document context.

10. Source References

The application displays the page numbers associated with retrieved chunks, allowing users to trace the answer back to the source document.

🛠️ Tech Stack

Component

Technology

Language

Python

UI

Streamlit

PDF Processing

PyMuPDF

Text Chunking

LangChain Text Splitters

Embeddings

sentence-transformers/all-MiniLM-L6-v2

Vector Search

FAISS

Lexical Search

BM25

Retrieval Fusion

Reciprocal Rank Fusion

Reranking

cross-encoder/ms-marco-MiniLM-L-6-v2

LLM

Qwen 2.5 7B

Local Inference

Ollama

📁 Project Structure

DocumentAssistant/
│
├── ui.py
├── main.py
├── rag.py
├── bm25.py
├── pdf_utils.py
├── config.py
├── requirements.txt
├── README.md
│
└── data/
    ├── raw_pdfs/
    └── faiss_index/

File

Purpose

ui.py

Streamlit interface, PDF upload and chat

main.py

Application initialization and FAISS index loading/building

rag.py

RAG pipeline, retrieval, RRF, reranking and LLM generation

bm25.py

BM25 lexical retrieval

pdf_utils.py

PDF extraction and chunking

config.py

Models, paths and retrieval configuration

requirements.txt

Python dependencies

⚙️ Configuration

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:7b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 5
SIMILARITY_THRESHOLD = 0.30

🚀 Installation

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/DocumentAssistant.git
cd DocumentAssistant

2. Create a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

macOS/Linux:

python3 -m venv venv
source venv/bin/activate

3. Install Python dependencies

pip install -r requirements.txt

4. Install Ollama

Install Ollama separately from:

https://ollama.com/

Then pull the model used by this project:

ollama pull qwen2.5:7b

Make sure Ollama is running before starting the application.

5. Run the application

python -m streamlit run ui.py

💬 Example Questions

After uploading a PDF:

What is an LLM?

Explain the RAG pipeline.

What is BM25?

Why is hybrid retrieval useful?

Compare BM25 and dense retrieval.

Why is reranking required?

What are the limitations discussed in the document?

The quality of the answer depends on the uploaded document and the retrieved context.

🎯 Why Hybrid Retrieval?

A major design choice in this project is combining semantic and lexical retrieval.

Dense retrieval

FAISS uses embeddings to find passages that are semantically similar to the query.

Lexical retrieval

BM25 focuses on term-level matching.

Combined approach

                 User Query
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       FAISS                  BM25
   semantic search        lexical search
          │                     │
          └──────────┬──────────┘
                     ▼
                    RRF
                     ▼
              Cross-Encoder
                     ▼
              Best passages

This gives the system complementary semantic and lexical retrieval signals.

⚡ Performance Considerations

The generated FAISS index and chunk metadata are stored locally:

data/faiss_index/
    ├── index.faiss
    └── chunks.pkl

When the cached index exists, it can be loaded instead of regenerating the embeddings.

When a new PDF is uploaded, the Streamlit interface clears the previous PDF/index and rebuilds the knowledge base for the new document.

⚠️ Current Limitations

The application is designed around one uploaded PDF at a time.

Retrieval quality depends on chunking and embedding quality.

Local LLM response speed depends on available CPU/GPU/RAM.

Scanned/image-only PDFs require OCR support, which is not currently implemented.

Complex questions may still produce imperfect retrieval or generation.

The current system is a document QA application rather than a general-purpose chatbot.

🔮 Future Improvements

Multi-document retrieval

Query rewriting and query expansion

Metadata-aware retrieval

Improved chunking strategies

OCR for scanned PDFs

Streaming LLM responses

Advanced citation highlighting

Retrieval evaluation using Precision@k and Recall@k

Faithfulness/grounding evaluation

Persistent document collections

Better multi-hop question handling

📌 RAG Concepts Demonstrated

PDF Processing
      ↓
Chunking
      ↓
Embeddings
      ↓
Vector Search
      ↓
Lexical Search
      ↓
Hybrid Retrieval
      ↓
RRF
      ↓
Reranking
      ↓
Context Construction
      ↓
LLM Generation
      ↓
Source Attribution

👩‍💻 Author

Prisha Singh

B.Tech, Computer Science and Engineering (Data Science)