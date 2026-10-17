# ============================================================
# app.py
# Conversational Research Paper Assistant
# Phase 1:
#   - PDF Upload
#   - arXiv Loading
#   - Dynamic Chunk Size / Chunk Overlap
#   - Conversational RAG
# ============================================================

# ------------------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES FIRST
# ------------------------------------------------------------

from dotenv import load_dotenv

load_dotenv()


# ------------------------------------------------------------
# 2. IMPORTS
# ------------------------------------------------------------

import os
import uuid
import streamlit as st

from src.loaders import load_pdf, load_arxiv
from src.splitter import split_documents

from src.vectorstore import (
    create_vectorstore,
    create_retriever
)

from src.rag_chain import build_rag_chain


# ------------------------------------------------------------
# 3. STREAMLIT PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)


# ------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------

def save_uploaded_pdf(uploaded_file):
    """
    Save Streamlit UploadedFile to the local uploads directory.

    PyPDFLoader works with a file path, so we first save the
    uploaded PDF locally.
    """

    os.makedirs(
        "data/uploads",
        exist_ok=True
    )

    # Prevent directory-related issues from uploaded filenames
    safe_filename = os.path.basename(
        uploaded_file.name
    )

    file_path = os.path.join(
        "data/uploads",
        safe_filename
    )

    with open(file_path, "wb") as file:
        file.write(
            uploaded_file.getbuffer()
        )

    return file_path


def create_new_session():
    """
    Create a new conversation ID.

    This is important when:
    - user changes paper
    - user clears chat

    Otherwise old LangChain conversation history could
    accidentally affect a new paper.
    """

    return str(uuid.uuid4())


def get_answer_text(answer):
    """
    Safely convert the RAG answer to displayable text.

    Usually our chain returns a string, but this also supports
    AIMessage-like objects.
    """

    if hasattr(answer, "content"):
        return answer.content

    return str(answer)


def get_source_label(doc, source_number):
    """
    Generate a readable label for a retrieved source.
    """

    metadata = doc.metadata or {}

    page = metadata.get("page")

    if isinstance(page, int):
        # PyPDFLoader normally uses zero-based page numbers
        return f"Source {source_number} — Page {page + 1}"

    title = (
        metadata.get("Title")
        or metadata.get("title")
    )

    if title:
        return f"Source {source_number} — {title}"

    return f"Source {source_number}"


# ------------------------------------------------------------
# 5. INITIALIZE STREAMLIT SESSION STATE
# ------------------------------------------------------------

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


if "messages" not in st.session_state:
    st.session_state.messages = []


if "paper_info" not in st.session_state:
    st.session_state.paper_info = None


if "session_id" not in st.session_state:
    st.session_state.session_id = create_new_session()


# ------------------------------------------------------------
# 6. HEADER
# ------------------------------------------------------------

st.title("📚 Conversational Research Paper Assistant")

st.caption(
    "Upload a research paper or load one from arXiv, "
    "then ask grounded questions about its content."
)


# ============================================================
# SIDEBAR
# ============================================================

# ------------------------------------------------------------
# 7. PAPER SOURCE
# ------------------------------------------------------------

st.sidebar.header("📄 Paper Source")

source_type = st.sidebar.radio(
    "Choose paper source",
    [
        "Upload PDF",
        "arXiv Paper"
    ]
)


uploaded_file = None
arxiv_id = None


if source_type == "Upload PDF":

    uploaded_file = st.sidebar.file_uploader(
        "Upload a research paper",
        type=["pdf"],
        help="Upload a text-readable PDF research paper."
    )


else:

    arxiv_id = st.sidebar.text_input(
        "Enter arXiv ID",
        placeholder="Example: 1706.03762",
        help=(
            "Enter an arXiv identifier. "
            "For example, Attention Is All You Need: 1706.03762"
        )
    )


# ------------------------------------------------------------
# 8. DOCUMENT PROCESSING SETTINGS
# ------------------------------------------------------------

st.sidebar.divider()

st.sidebar.header("⚙️ Document Processing")


chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=300,
    max_value=2000,
    value=1000,
    step=100,
    help=(
        "Maximum approximate size of each text chunk "
        "sent into the embedding pipeline."
    )
)


chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=500,
    value=150,
    step=25,
    help=(
        "Amount of overlapping text between neighboring chunks."
    )
)


if chunk_overlap >= chunk_size:

    st.sidebar.warning(
        "⚠️ Chunk overlap must be smaller than chunk size."
    )


# ------------------------------------------------------------
# 9. PROCESS PAPER BUTTON
# ------------------------------------------------------------

process_button = st.sidebar.button(
    "🚀 Process Paper",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PAPER PROCESSING
# ============================================================

if process_button:

    # --------------------------------------------------------
    # Validate chunk settings
    # --------------------------------------------------------

    if chunk_overlap >= chunk_size:

        st.error(
            "Chunk overlap must be smaller than chunk size."
        )

    else:

        try:

            with st.spinner(
                "Processing research paper..."
            ):

                # ============================================
                # STEP 1 — LOAD DOCUMENT
                # ============================================

                if source_type == "Upload PDF":

                    if uploaded_file is None:

                        raise ValueError(
                            "Please upload a PDF before processing."
                        )

                    pdf_path = save_uploaded_pdf(
                        uploaded_file
                    )

                    documents = load_pdf(
                        pdf_path
                    )

                    paper_name = uploaded_file.name


                else:

                    if not arxiv_id or not arxiv_id.strip():

                        raise ValueError(
                            "Please enter an arXiv paper ID."
                        )

                    documents = load_arxiv(
                        arxiv_id.strip()
                    )

                    metadata = (
                        documents[0].metadata
                        if documents
                        else {}
                    )

                    paper_name = (
                        metadata.get("Title")
                        or metadata.get("title")
                        or arxiv_id.strip()
                    )


                # ============================================
                # STEP 2 — SPLIT DOCUMENTS
                # ============================================

                chunks = split_documents(
                    documents=documents,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )


                # ============================================
                # STEP 3 — CREATE EMBEDDINGS + FAISS
                # ============================================

                vector_store = create_vectorstore(
                    chunks
                )


                # ============================================
                # STEP 4 — CREATE RETRIEVER
                # ============================================

                retriever = create_retriever(
                    vector_store
                )


                # ============================================
                # STEP 5 — BUILD CONVERSATIONAL RAG
                # ============================================

                rag_chain = build_rag_chain(
                    retriever
                )


                # ============================================
                # STEP 6 — STORE IN STREAMLIT STATE
                # ============================================

                st.session_state.rag_chain = rag_chain

                st.session_state.paper_info = {
                    "name": paper_name,
                    "source": source_type,
                    "documents": len(documents),
                    "chunks": len(chunks),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }


                # New paper = new conversation
                st.session_state.messages = []

                st.session_state.session_id = (
                    create_new_session()
                )


            st.success(
                "✅ Paper processed successfully. "
                "You can now ask questions."
            )


        except Exception as error:

            st.error(
                f"❌ Failed to process paper: {error}"
            )


# ============================================================
# ACTIVE PAPER INFORMATION
# ============================================================

if st.session_state.paper_info:

    info = st.session_state.paper_info

    st.sidebar.divider()

    st.sidebar.subheader(
        "📚 Active Paper"
    )

    st.sidebar.write(
        f"**Name:** {info['name']}"
    )

    st.sidebar.write(
        f"**Source:** {info['source']}"
    )

    st.sidebar.write(
        f"**Documents / Pages:** {info['documents']}"
    )

    st.sidebar.write(
        f"**Chunks:** {info['chunks']}"
    )

    st.sidebar.write(
        f"**Chunk Size:** {info['chunk_size']}"
    )

    st.sidebar.write(
        f"**Chunk Overlap:** {info['chunk_overlap']}"
    )


# ============================================================
# CLEAR CHAT
# ============================================================

if st.session_state.rag_chain is not None:

    st.sidebar.divider()

    if st.sidebar.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        # Clear Streamlit UI messages
        st.session_state.messages = []

        # Start a fresh LangChain conversation
        # while keeping the current paper/index
        st.session_state.session_id = (
            create_new_session()
        )

        st.rerun()


# ============================================================
# MAIN CHAT AREA
# ============================================================

# ------------------------------------------------------------
# 10. NO PAPER PROCESSED YET
# ------------------------------------------------------------

if st.session_state.rag_chain is None:

    st.info(
        "👈 Upload a PDF or enter an arXiv paper ID "
        "from the sidebar, configure the document settings, "
        "and click **Process Paper**."
    )

    st.markdown(
        """
### How it works

1. Select **Upload PDF** or **arXiv Paper**
2. Choose chunk size and overlap
3. Click **Process Paper**
4. Ask questions about the research paper
5. Ask follow-up questions naturally
6. Inspect the retrieved sources
        """
    )


# ------------------------------------------------------------
# 11. PAPER READY — DISPLAY CHAT
# ------------------------------------------------------------

else:

    paper_name = (
        st.session_state.paper_info["name"]
    )

    st.subheader(
        f"💬 Chat with: {paper_name}"
    )


    # --------------------------------------------------------
    # Display previous messages
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            # Redisplay saved sources for assistant messages
            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander(
                    "📖 View Sources"
                ):

                    for index, doc in enumerate(
                        message["sources"],
                        start=1
                    ):

                        st.markdown(
                            f"**{get_source_label(doc, index)}**"
                        )

                        st.write(
                            doc.page_content[:700]
                        )

                        st.divider()


    # --------------------------------------------------------
    # 12. CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about the research paper..."
    )


    if question:

        # ----------------------------------------------------
        # Save and display user message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        with st.chat_message("user"):

            st.markdown(
                question
            )


        # ----------------------------------------------------
        # Invoke conversational RAG
        # ----------------------------------------------------

        try:

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Searching the paper..."
                ):

                    result = (
                        st.session_state
                        .rag_chain
                        .invoke(
                            {
                                "question": question
                            },
                            config={
                                "configurable": {
                                    "session_id":
                                        st.session_state.session_id
                                }
                            }
                        )
                    )


                # --------------------------------------------
                # Handle dictionary output
                # --------------------------------------------

                if isinstance(result, dict):

                    answer = result.get(
                        "answer",
                        "I could not generate an answer."
                    )

                    sources = result.get(
                        "sources",
                        []
                    )

                else:

                    answer = result
                    sources = []


                answer_text = get_answer_text(
                    answer
                )


                # --------------------------------------------
                # Display answer
                # --------------------------------------------

                st.markdown(
                    answer_text
                )


                # --------------------------------------------
                # Display retrieved sources
                # --------------------------------------------

                if sources:

                    with st.expander(
                        "📖 View Sources"
                    ):

                        for index, doc in enumerate(
                            sources,
                            start=1
                        ):

                            st.markdown(
                                f"**{get_source_label(doc, index)}**"
                            )

                            st.write(
                                doc.page_content[:700]
                            )

                            st.divider()


            # ------------------------------------------------
            # Save assistant response
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources
                }
            )


        except Exception as error:

            st.error(
                f"❌ Error while answering the question: {error}"
            )