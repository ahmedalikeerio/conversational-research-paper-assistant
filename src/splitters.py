from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(
        documents,
        chunk_size,
        chunk_overlap
):
    """ Split langchain documents into chunks"""

    if not documents:
        raise ValueError("Can't split documents because documents list is empty")

    if chunk_overlap>chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size")

    splitter = RecursiveCharacterTextSplitter()
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created from the documents")

    return chunks