# ============================================================
# app.py
# Conversational Research Paper Assistant
#
# Phase 1
#   - PDF Upload
#   - arXiv Paper Loading
#   - Dynamic Chunk Size
#   - Dynamic Chunk Overlap
#
# Phase 2
#   - Dynamic Retriever k
#   - Similarity / MMR Retrieval
#   - Dynamic LLM Temperature
#   - Active RAG Configuration
#
# Phase 3
#   - History-Aware Conversational Retrieval
#   - Standalone Question Rewriting
#   - RAG Debug Panel
#   - Original vs Rewritten Question Inspection
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

from src.loaders import (
    load_pdf,
    load_arxiv
)

from src.splitters import (
    split_documents
)

from src.vectorstore import (
    create_vectorstore,
    create_retriever
)

from src.rag_chain import (
    build_rag_chain
)


# ------------------------------------------------------------
# 3. STREAMLIT PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def save_uploaded_pdf(uploaded_file):
    """
    Save a Streamlit UploadedFile locally so PyPDFLoader
    can read it using a normal file path.
    """

    os.makedirs(
        "data/uploads",
        exist_ok=True
    )

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
    Create a unique LangChain conversation session.

    A new session is created when:
    - a new paper is processed
    - the user clears the chat
    """

    return str(
        uuid.uuid4()
    )


def get_answer_text(answer):
    """
    Convert either a string or an AIMessage-like
    object into displayable text.
    """

    if hasattr(answer, "content"):
        return answer.content

    return str(answer)


def get_source_label(
    doc,
    source_number
):
    """
    Create a human-readable label for retrieved sources.

    Local PDFs usually contain page metadata.
    arXiv documents may instead contain title metadata.
    """

    metadata = (
        doc.metadata
        if doc.metadata
        else {}
    )

    page = metadata.get(
        "page"
    )

    if isinstance(page, int):

        return (
            f"Source {source_number} "
            f"— Page {page + 1}"
        )

    title = (
        metadata.get("Title")
        or metadata.get("title")
    )

    if title:

        return (
            f"Source {source_number} "
            f"— {title}"
        )

    return (
        f"Source {source_number}"
    )


def display_sources(sources):
    """
    Display retrieved chunks inside Streamlit.
    """

    if not sources:
        return

    with st.expander(
        "📖 View Sources"
    ):

        for index, doc in enumerate(
            sources,
            start=1
        ):

            label = get_source_label(
                doc,
                index
            )

            st.markdown(
                f"**{label}**"
            )

            st.write(
                doc.page_content[:700]
            )

            st.divider()


def display_debug_info(
    original_question,
    standalone_question,
    sources
):
    """
    Display conversational RAG internals.
    """

    with st.expander(
        "🛠️ RAG Debug Information"
    ):

        # ---------------------------------------------
        # Original user question
        # ---------------------------------------------

        st.markdown(
            "### Original Question"
        )

        st.code(
            original_question
        )

        # ---------------------------------------------
        # Question actually sent to retriever
        # ---------------------------------------------

        st.markdown(
            "### Retrieval Question"
        )

        st.code(
            standalone_question
        )

        # ---------------------------------------------
        # Retrieval information
        # ---------------------------------------------

        st.markdown(
            "### Retrieval Details"
        )

        st.write(
            f"**Retrieved chunks:** "
            f"{len(sources)}"
        )

        if st.session_state.paper_info:

            info = (
                st.session_state.paper_info
            )

            st.write(
                f"**Search type:** "
                f"{info['search_type']}"
            )

            st.write(
                f"**k:** "
                f"{info['k']}"
            )

            st.write(
                f"**Chunk size:** "
                f"{info['chunk_size']}"
            )

            st.write(
                f"**Chunk overlap:** "
                f"{info['chunk_overlap']}"
            )

            st.write(
                f"**Temperature:** "
                f"{info['temperature']}"
            )


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================


if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


if "messages" not in st.session_state:
    st.session_state.messages = []


if "paper_info" not in st.session_state:
    st.session_state.paper_info = None


if "session_id" not in st.session_state:
    st.session_state.session_id = (
        create_new_session()
    )


# ============================================================
# MAIN HEADER
# ============================================================


st.title(
    "📚 Conversational Research Paper Assistant"
)


st.caption(
    "Upload a research paper or load one from arXiv, "
    "configure the RAG pipeline, and ask grounded "
    "conversational questions."
)


# ============================================================
# SIDEBAR — PAPER SOURCE
# ============================================================


st.sidebar.header(
    "📄 Paper Source"
)


source_type = st.sidebar.radio(
    "Choose paper source",
    [
        "Upload PDF",
        "arXiv Paper"
    ]
)


uploaded_file = None
arxiv_id = None


# ------------------------------------------------------------
# PDF Source
# ------------------------------------------------------------

if source_type == "Upload PDF":

    uploaded_file = (
        st.sidebar.file_uploader(
            "Upload a research paper",
            type=["pdf"],
            help=(
                "Upload a PDF containing "
                "extractable text."
            )
        )
    )


# ------------------------------------------------------------
# arXiv Source
# ------------------------------------------------------------

else:

    arxiv_id = (
        st.sidebar.text_input(
            "Enter arXiv ID",
            placeholder="Example: 1706.03762",
            help=(
                "Enter the arXiv identifier "
                "of the research paper."
            )
        )
    )


# ============================================================
# SIDEBAR — CHUNKING SETTINGS
# ============================================================


st.sidebar.divider()


st.sidebar.header(
    "✂️ Chunking Settings"
)


chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=300,
    max_value=2000,
    value=1000,
    step=100,
    help=(
        "Controls the approximate maximum size "
        "of each text chunk."
    )
)


chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=500,
    value=150,
    step=25,
    help=(
        "Controls how much neighboring chunks "
        "overlap."
    )
)


if chunk_overlap >= chunk_size:

    st.sidebar.warning(
        "⚠️ Chunk overlap must be smaller "
        "than chunk size."
    )


# ============================================================
# SIDEBAR — RETRIEVAL SETTINGS
# ============================================================


st.sidebar.divider()


st.sidebar.header(
    "🔎 Retrieval Settings"
)


k = st.sidebar.slider(
    "Retrieved Chunks (k)",
    min_value=1,
    max_value=10,
    value=4,
    step=1,
    help=(
        "Number of chunks retrieved from "
        "the vector store for each question."
    )
)


search_type = st.sidebar.selectbox(
    "Retrieval Strategy",
    options=[
        "similarity",
        "mmr"
    ],
    index=0,
    help=(
        "Similarity returns the nearest chunks. "
        "MMR balances relevance and diversity."
    )
)


# ============================================================
# SIDEBAR — MODEL SETTINGS
# ============================================================


st.sidebar.divider()


st.sidebar.header(
    "🤖 Model Settings"
)


temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    help=(
        "Controls randomness in LLM generation. "
        "Lower values are generally preferable "
        "for grounded research QA."
    )
)


# ============================================================
# PHASE 3 — DEVELOPER TOOLS
# ============================================================


st.sidebar.divider()


st.sidebar.header(
    "🛠️ Developer Tools"
)


show_debug = st.sidebar.toggle(
    "Show RAG Debug Information",
    value=False,
    help=(
        "Inspect the original question, "
        "history-aware rewritten retrieval question, "
        "and retrieval configuration."
    )
)


st.sidebar.caption(
    "RAG configuration changes take effect "
    "after clicking **Process Paper**."
)


# ============================================================
# PROCESS PAPER BUTTON
# ============================================================


process_button = st.sidebar.button(
    "🚀 Process Paper",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PAPER PROCESSING PIPELINE
# ============================================================


if process_button:

    # --------------------------------------------------------
    # Validate chunk configuration
    # --------------------------------------------------------

    if chunk_overlap >= chunk_size:

        st.error(
            "❌ Chunk overlap must be smaller "
            "than chunk size."
        )

    else:

        try:

            with st.spinner(
                "Processing research paper..."
            ):

                # =================================================
                # STEP 1 — LOAD DOCUMENT
                # =================================================

                if source_type == "Upload PDF":

                    if uploaded_file is None:

                        raise ValueError(
                            "Please upload a PDF "
                            "before processing."
                        )

                    pdf_path = (
                        save_uploaded_pdf(
                            uploaded_file
                        )
                    )

                    documents = (
                        load_pdf(
                            pdf_path
                        )
                    )

                    paper_name = (
                        uploaded_file.name
                    )

                # =================================================
                # STEP 1B — LOAD arXiv PAPER
                # =================================================

                else:

                    if (
                        not arxiv_id
                        or not arxiv_id.strip()
                    ):

                        raise ValueError(
                            "Please enter an arXiv ID."
                        )

                    documents = (
                        load_arxiv(
                            arxiv_id.strip()
                        )
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

                # =================================================
                # STEP 2 — SPLIT DOCUMENTS
                # =================================================

                chunks = (
                    split_documents(
                        documents=documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                )

                # =================================================
                # STEP 3 — CREATE EMBEDDINGS + FAISS
                # =================================================

                vector_store = (
                    create_vectorstore(
                        chunks
                    )
                )

                # =================================================
                # STEP 4 — SAFE RETRIEVAL k
                # =================================================

                effective_k = min(
                    k,
                    len(chunks)
                )

                # =================================================
                # STEP 5 — CREATE RETRIEVER
                # =================================================

                retriever = (
                    create_retriever(
                        vector_store=vector_store,
                        k=effective_k,
                        search_type=search_type
                    )
                )

                # =================================================
                # STEP 6 — BUILD CONVERSATIONAL RAG
                # =================================================

                rag_chain = (
                    build_rag_chain(
                        retriever=retriever,
                        temperature=temperature
                    )
                )

                # =================================================
                # STEP 7 — SAVE ACTIVE RAG CHAIN
                # =================================================

                st.session_state.rag_chain = (
                    rag_chain
                )

                # =================================================
                # STEP 8 — SAVE PAPER / RAG INFORMATION
                # =================================================

                st.session_state.paper_info = {

                    "name":
                        paper_name,

                    "source":
                        source_type,

                    "documents":
                        len(documents),

                    "chunks":
                        len(chunks),

                    "chunk_size":
                        chunk_size,

                    "chunk_overlap":
                        chunk_overlap,

                    "k":
                        effective_k,

                    "search_type":
                        search_type,

                    "temperature":
                        temperature
                }

                # =================================================
                # STEP 9 — RESET OLD CONVERSATION
                # =================================================

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
# SIDEBAR — ACTIVE PAPER INFO
# ============================================================


if st.session_state.paper_info:

    info = (
        st.session_state.paper_info
    )

    st.sidebar.divider()


    # --------------------------------------------------------
    # Active Paper
    # --------------------------------------------------------

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
        f"**Documents / Pages:** "
        f"{info['documents']}"
    )


    st.sidebar.write(
        f"**Chunks Created:** "
        f"{info['chunks']}"
    )


    # --------------------------------------------------------
    # Active RAG Configuration
    # --------------------------------------------------------

    st.sidebar.divider()


    st.sidebar.subheader(
        "🧠 Active RAG Config"
    )


    st.sidebar.write(
        f"**Chunk Size:** "
        f"{info['chunk_size']}"
    )


    st.sidebar.write(
        f"**Chunk Overlap:** "
        f"{info['chunk_overlap']}"
    )


    st.sidebar.write(
        f"**Retriever k:** "
        f"{info['k']}"
    )


    st.sidebar.write(
        f"**Search Type:** "
        f"{info['search_type']}"
    )


    st.sidebar.write(
        f"**Temperature:** "
        f"{info['temperature']}"
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

        # ---------------------------------------------
        # Clear visible Streamlit messages
        # ---------------------------------------------

        st.session_state.messages = []

        # ---------------------------------------------
        # Create fresh LangChain history
        # while keeping same processed paper
        # ---------------------------------------------

        st.session_state.session_id = (
            create_new_session()
        )

        st.rerun()


# ============================================================
# MAIN CONTENT — NO PAPER ACTIVE
# ============================================================


if st.session_state.rag_chain is None:

    st.info(
        "👈 Upload a PDF or enter an arXiv paper ID, "
        "configure your RAG settings, and click "
        "**Process Paper**."
    )


    st.markdown(
        """
### 🚀 How to use the assistant

1. Choose **Upload PDF** or **arXiv Paper**
2. Configure chunk size and overlap
3. Choose retrieval `k`
4. Select **Similarity** or **MMR**
5. Select LLM temperature
6. Click **Process Paper**
7. Ask a question about the paper
8. Ask conversational follow-up questions
9. Inspect retrieved sources
10. Enable **RAG Debug Information** to inspect the retrieval query
        """
    )


# ============================================================
# MAIN CONTENT — PAPER ACTIVE
# ============================================================


else:

    paper_name = (
        st.session_state
        .paper_info["name"]
    )


    st.subheader(
        f"💬 Chat with: {paper_name}"
    )


    # ========================================================
    # DISPLAY PREVIOUS CHAT MESSAGES
    # ========================================================

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


            # ------------------------------------------------
            # Display previous sources
            # ------------------------------------------------

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                display_sources(
                    message["sources"]
                )


            # ------------------------------------------------
            # Display previously saved debug info
            # ------------------------------------------------

            if (
                show_debug
                and message["role"] == "assistant"
            ):

                original_question = (
                    message.get(
                        "original_question"
                    )
                )

                standalone_question = (
                    message.get(
                        "standalone_question"
                    )
                )

                if (
                    original_question
                    and standalone_question
                ):

                    display_debug_info(
                        original_question=
                            original_question,

                        standalone_question=
                            standalone_question,

                        sources=
                            message.get(
                                "sources",
                                []
                            )
                    )


    # ========================================================
    # USER CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask a question about the research paper..."
    )


    # ========================================================
    # HANDLE NEW QUESTION
    # ========================================================

    if question:

        # ----------------------------------------------------
        # Save user question to Streamlit UI history
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role":
                    "user",

                "content":
                    question
            }
        )


        # ----------------------------------------------------
        # Display user question immediately
        # ----------------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )


        # ====================================================
        # EXECUTE RAG PIPELINE
        # ====================================================

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
                                "question":
                                    question
                            },
                            config={
                                "configurable": {
                                    "session_id":
                                        st.session_state
                                        .session_id
                                }
                            }
                        )
                    )


                # =================================================
                # READ STRUCTURED RAG RESULT
                # =================================================

                if isinstance(
                    result,
                    dict
                ):

                    answer = (
                        result.get(
                            "answer",
                            "I could not generate an answer."
                        )
                    )


                    sources = (
                        result.get(
                            "sources",
                            []
                        )
                    )


                    original_question = (
                        result.get(
                            "original_question",
                            question
                        )
                    )


                    standalone_question = (
                        result.get(
                            "standalone_question",
                            question
                        )
                    )


                else:

                    # Fallback if chain unexpectedly returns text
                    answer = result

                    sources = []

                    original_question = (
                        question
                    )

                    standalone_question = (
                        question
                    )


                # =================================================
                # CONVERT ANSWER TO TEXT
                # =================================================

                answer_text = (
                    get_answer_text(
                        answer
                    )
                )


                # =================================================
                # DISPLAY ANSWER
                # =================================================

                st.markdown(
                    answer_text
                )


                # =================================================
                # DISPLAY SOURCES
                # =================================================

                display_sources(
                    sources
                )


                # =================================================
                # PHASE 3 — DISPLAY DEBUG INFO
                # =================================================

                if show_debug:

                    display_debug_info(
                        original_question=
                            original_question,

                        standalone_question=
                            standalone_question,

                        sources=
                            sources
                    )


            # =====================================================
            # SAVE ASSISTANT RESPONSE IN STREAMLIT HISTORY
            # =====================================================

            st.session_state.messages.append(
                {
                    "role":
                        "assistant",

                    "content":
                        answer_text,

                    "sources":
                        sources,

                    "original_question":
                        original_question,

                    "standalone_question":
                        standalone_question
                }
            )


        # ========================================================
        # HANDLE QUERY ERROR
        # ========================================================

        except Exception as error:

            st.error(
                f"❌ Error while answering "
                f"the question: {error}"
            )