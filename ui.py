import shutil
from pathlib import Path

import streamlit as st

from config import RAW_PDF_DIR, FAISS_DIR
from main import initialize_chatbot


# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="DocumentAssistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocumentAssistant")
st.caption("Ask questions about your uploaded PDF using RAG.")


# ----------------------------------------------------
# Session State
# ----------------------------------------------------

if "assistant" not in st.session_state:
    st.session_state.assistant = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "loaded_pdf" not in st.session_state:
    st.session_state.loaded_pdf = None


# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

with st.sidebar:

    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    st.divider()

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        if st.session_state.assistant is not None:

            try:
                st.session_state.assistant.memory.clear()
            except:
                pass

        st.rerun()

    st.divider()


# ----------------------------------------------------
# Load PDF
# ----------------------------------------------------

if uploaded_file is not None:

    pdf_path = RAW_PDF_DIR / uploaded_file.name

    if st.session_state.loaded_pdf != uploaded_file.name:

        RAW_PDF_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        FAISS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # Remove old PDFs

        for file in RAW_PDF_DIR.glob("*.pdf"):
            file.unlink()

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Remove cached FAISS index

        for file in FAISS_DIR.glob("*"):
            file.unlink()

        st.session_state.loaded_pdf = uploaded_file.name

        st.session_state.messages = []

        with st.spinner("Building knowledge base..."):

            st.session_state.assistant = initialize_chatbot()

        st.success("PDF loaded successfully!")

# ----------------------------------------------------
# Initialize Assistant
# ----------------------------------------------------

if (
    st.session_state.assistant is None
    and st.session_state.loaded_pdf is not None
):

    with st.spinner("Loading Document Assistant..."):

        st.session_state.assistant = initialize_chatbot()


# ----------------------------------------------------
# Chat History
# ----------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and "sources" in message
            and message["sources"]
        ):

            pages = sorted(
                set(
                    chunk["page"]
                    for chunk in message["sources"]
                )
            )

            st.caption(
                "📚 Sources: "
                + ", ".join(
                    f"Page {page}"
                    for page in pages
                )
            )


# ----------------------------------------------------
# Chat Input
# ----------------------------------------------------

prompt = st.chat_input(
    "Ask a question about the uploaded PDF..."
)

if prompt:

    # ------------------------
    # Show User Message
    # ------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    # ------------------------
    # Check PDF
    # ------------------------

    if st.session_state.assistant is None:

        with st.chat_message("assistant"):

            st.warning(
                "Please upload a PDF first."
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Please upload a PDF first."
            }
        )

        st.stop()

    # ------------------------
    # Generate Answer
    # ------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                answer, sources = (
                    st.session_state.assistant.ask(prompt)
                )

                st.markdown(answer)

                pages = sorted(
                    set(
                        chunk["page"]
                        for chunk in sources
                    )
                )

                if pages:

                    st.caption(
                        "📚 Sources: "
                        + ", ".join(
                            f"Page {page}"
                            for page in pages
                        )
                    )

            except Exception as e:

                answer = (
                    "❌ An unexpected error occurred.\n\n"
                    f"```{e}```"
                )

                sources = []

                st.error(answer)

    # ------------------------
    # Save Assistant Message
    # ------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.divider()

