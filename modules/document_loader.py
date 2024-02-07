from langchain.document_loaders import PyPDFLoader

def load_pdf_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()
