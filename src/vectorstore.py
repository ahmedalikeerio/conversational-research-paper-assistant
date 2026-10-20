from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# function to generate embeddings and store it into the FAISS vector store
def create_vectorstore(chunks):
    # generate embediings

    if not chunks:
        raise ValueError("Can't create vector store because chunka are not created")
    
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

def create_retriever(vector_store,
                    k=4,
                    search_type = 'similarity'):


    if k<=0:
        raise ValueError(
            "k must be greater than zero"
        )

    if search_type not in [
        "similarity",
        'mmr'
    ]:
        raise ValueError(
            "search_type shoule be either similarity or mmr."
        )

    if search_type == "similarity":
        return vector_store.as_retriever(
            search_type = "similarity",
            search_kwargs = {
                "k":k
            }
        )

    if search_type == 'mmr':
        return vector_store.as_retriever(
            search_type = 'mmr',
            search_kwargs ={
                "k":k,
                "fetch_k": max(k*3,10)
            }

        )
