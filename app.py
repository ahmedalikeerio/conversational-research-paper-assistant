import streamlit as st

from dotenv import load_dotenv

load_dotenv()

from src.loader import (
    load_and_split_pdf
)

from src.vectorstore import (
    create_vectorstore,
    create_retriever
)

from src.rag_chain import (
    build_rag_chain
)



st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📄",
    layout="wide"
)


st.title(
    "📄 Conversational Research Paper Assistant"
)

st.caption(
    "Ask questions about Attention Is All You Need."
)


pdf_path = (
    "data/attention_is_all_you_need.pdf"
)


@st.cache_resource
def initialize_rag():

    chunks = load_and_split_pdf(
        pdf_path
    )

    vector_store = create_vectorstore(
        chunks
    )

    retriever = create_retriever(
        vector_store
    )

    return build_rag_chain(
        retriever
    )


rag_chain = initialize_rag()


if "messages" not in st.session_state:

    st.session_state.messages = []


with st.sidebar:

    st.header(
        "Settings"
    )

    st.write(
        "Paper: Attention Is All You Need"
    )

    if st.button(
        "Clear Conversation"
    ):

        st.session_state.messages = []

        st.rerun()


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


question = st.chat_input(
    "Ask a question about the paper..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    with st.chat_message("user"):

        st.markdown(question)


    result = rag_chain.invoke(
        {
            "question": question
        },
        config={
            "configurable": {
                "session_id":
                "streamlit_user"
            }
        }
    )


    answer = result["answer"]


    with st.chat_message(
        "assistant"
    ):

        st.markdown(answer)

        with st.expander(
            "View retrieved sources"
        ):

            for i, doc in enumerate(
                result["sources"],
                start=1
            ):

                page = (
                    doc.metadata.get(
                        "page",
                        "Unknown"
                    )
                )

                st.markdown(
                    f"### Source {i} — Page {page}"
                )

                st.write(
                    doc.page_content
                )


    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })