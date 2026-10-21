from pathlib import Path

import streamlit as st

from config import RAW_PDF_DIR
from rag_pipeline import DocumentAssistant


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DocumentAssistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocumentAssistant")
st.caption("Ask questions about your PDF using hybrid RAG.")


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "assistant" not in st.session_state:
    st.session_state.assistant = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "loaded_pdf_id" not in st.session_state:
    st.session_state.loaded_pdf_id = None


# --------------------------------------------------
# Sidebar: Upload PDF
# --------------------------------------------------

with st.sidebar:
    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
    )

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "Powered by LangChain, FAISS, BM25, "
        "Cross-Encoder Reranking and Ollama."
    )


# --------------------------------------------------
# Load Uploaded PDF
# --------------------------------------------------

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    # Identify the uploaded file by its name and content.
    import hashlib

    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    pdf_id = f"{uploaded_file.name}:{file_hash}"

    if st.session_state.loaded_pdf_id != pdf_id:

        RAW_PDF_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save the uploaded PDF.
        pdf_path = RAW_PDF_DIR / Path(
            uploaded_file.name
        ).name

        with open(pdf_path, "wb") as file:
            file.write(pdf_bytes)

        # Build a fresh assistant for this PDF.
        with st.spinner(
            "Building your knowledge base... "
            "This may take a few minutes."
        ):

            try:
                assistant = DocumentAssistant()

                assistant.load_pdf(pdf_path)

                st.session_state.assistant = assistant
                st.session_state.loaded_pdf_id = pdf_id
                st.session_state.messages = []

                st.success(
                    f"Successfully loaded: {uploaded_file.name}"
                )

            except Exception as error:

                st.session_state.assistant = None
                st.session_state.loaded_pdf_id = None

                st.error(
                    f"Could not load the PDF: {error}"
                )


# --------------------------------------------------
# Display Uploaded PDF Status
# --------------------------------------------------

if st.session_state.assistant is not None:

    st.success("Your PDF is ready. Ask away!")

else:

    st.info(
        "Upload a PDF using the sidebar to get started."
    )


# --------------------------------------------------
# Display Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display citations for assistant responses.
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 View Sources"):

                for index, source in enumerate(
                    message["sources"],
                    start=1,
                ):

                    st.markdown(
                        f"**Source {index}: "
                        f"{source.get('source', 'Unknown')}**"
                    )

                    st.caption(
                        f"Page {source.get('page', 'Unknown')}"
                    )

                    st.write(
                        source.get("content", "")
                    )

                    st.divider()


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your PDF..."
)


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------

if question:

    # Save and display the user's question.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Ensure a PDF has been loaded.
    if st.session_state.assistant is None:

        st.warning(
            "Please upload a PDF before asking questions."
        )

    else:

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching your document and generating an answer..."
            ):

                try:
                    # The new pipeline returns a dictionary.
                    result = st.session_state.assistant.ask(
                        question
                    )

                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    # Display source citations.
                    if sources:

                        with st.expander("📚 View Sources"):

                            for index, source in enumerate(
                                sources,
                                start=1,
                            ):

                                st.markdown(
                                    f"**Source {index}: "
                                    f"{source.get('source', 'Unknown')}**"
                                )

                                st.caption(
                                    f"Page {source.get('page', 'Unknown')}"
                                )

                                st.write(
                                    source.get("content", "")
                                )

                                st.divider()

                    else:

                        st.caption(
                            "No supporting source passages were retrieved."
                        )

                    # Save the answer and citations.
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as error:

                    st.error(
                        f"An error occurred while answering: {error}"
                    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "DocumentAssistant | Hybrid Retrieval-Augmented Generation"
)