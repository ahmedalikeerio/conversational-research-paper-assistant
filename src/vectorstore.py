from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# function to generate embeddings and store it into the FAISS vector store
def create_vectorstore(chunks):
    # generate embediings
    embeddings = OllamaEmbeddings(
        model='nomic-embed-text'
    )

    # store embeddings into vector store
    vector_store = FAISS.from_documents(
        embedding=embeddings,
        documents=chunks
    )

    # returning the vector store
    return vector_store

def create_retriever(vector_store):
    return vector_store.as_retriever(
        search_kwargs = {
            "k":'4'
        }
    )