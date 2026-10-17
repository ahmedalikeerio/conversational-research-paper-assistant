from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# function to load the pdf and split it into chunks
def load_and_split_pdf(pdf_path):

    # load the pdf using path into documents
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print("Documents loaded:", len(documents))

    for i, doc in enumerate(documents):
        print(i, len(doc.page_content))

    # split the pdf into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )

    chunks = splitter.split_documents(documents)
    print("Chunks created:", len(chunks))

    return chunks
