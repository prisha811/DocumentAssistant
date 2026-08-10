from pathlib import Path

from rag import (
    DocumentAssistant,
    build_index,
    save_index,
    load_index,
    cache_exists,
)

from pdf_utils import (
    load_pdf,
    chunk_pages,
)

from config import (
    BASE_DIR,
    DATA_DIR,
    RAW_PDF_DIR,
)


def initialize_chatbot():

    pdf_files = list(Path(RAW_PDF_DIR).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF found inside data/raw_pdfs"
        )

    pdf_path = pdf_files[0]

    if cache_exists():

        print("✅ Loading cached FAISS index...")

        index, chunks = load_index()

    else:

        print("📄 Reading PDF...")

        pages = load_pdf(pdf_path)

        print(f"Loaded {len(pages)} pages")

        print("✂️ Chunking...")

        chunks = chunk_pages(pages)

        print(f"Created {len(chunks)} chunks")

        print("🧠 Creating embeddings & FAISS index...")

        index = build_index(chunks)

        save_index(index, chunks)

        print("✅ Index saved")

    return DocumentAssistant(index, chunks)


def main():

    chatbot = initialize_chatbot()

    print("\n" + "=" * 35)
    print("📄 DocumentAssistant")
    print("=" * 35)
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            print("\n👋 Goodbye!")
            break

        answer, sources = chatbot.ask(question)

        print("\nDocumentAssistant:\n")
        print(answer)

        pages = sorted(
            set(chunk["page"] for chunk in sources)
        )

        if pages:
            print("\n📚 Sources")
            print(", ".join(f"Page {page}" for page in pages))

        print()


if __name__ == "__main__":
    main()