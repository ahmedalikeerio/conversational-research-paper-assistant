from langchain_community.document_loaders import PyPDFLoader, ArxivLoader

# function to load the pf
def load_pdf(pdf_path):
    """
    load an pdf and return langchain documents
    """
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    if not documents:
        raise ValueError("No pages were loaded")

    total_text = sum(len(doc.page_content.strip()) for doc in documents)
 
    if total_text == 0:
        raise ValueError("Pdf was opened, but no readable text could be extracted")

    return documents

# function to laod the pdf using arxiv id
def load_arxiv(arxiv_id):
    """ load an paper using arxiv id """

    if not arxiv_id.strip():
        raise ValueError("Please enter an valid Arxiv ID")

    loader = ArxivLoader(
        query=arxiv_id.strip(),
        load_max_docs =1
    )

    documents = loader.load()

    if not documents:
        raise ValueError(f"No paper found using the Arxiv Id: {arxiv_id}")

    total_text = sum(len(docs.page_content.strip()) for docs in documents)

    if total_text == 0:
        raise ValueError(
            "The arxiv document was opened, but no readable text could be extracted"
        )

    return documents
