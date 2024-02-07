from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def split_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(data)

def generate_embedding(text):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return model.encode(text)
