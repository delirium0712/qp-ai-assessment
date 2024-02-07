import logging
import warnings
from modules.document_loader import load_pdf_document
from modules.text_processing import split_documents, generate_embedding
from modules.milvus_operations import create_milvus_collection, insert_documents_to_collection, search_documents_in_collection
from modules.chat_bot import search_and_answer_question

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Ignore warnings
warnings.filterwarnings("ignore")

def main():
    pdf_path = "attention-is-all-you-need-Paper.pdf"
    data = load_pdf_document(pdf_path)
    
    texts = split_documents(data)
    embeddings = [generate_embedding(text.page_content) for text in texts[:10]]
    texts_list = [text.page_content for text in texts]
    
    collection_name = "search_article_in_medium_v2"
    collection = create_milvus_collection(collection_name, 768)
    num_entities = insert_documents_to_collection(collection_name, texts_list, embeddings)
    logging.info(f"Inserted {num_entities} documents into collection {collection_name}")
    
    query = "What is a Sentence Transformer"
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    
    results = search_documents_in_collection(collection_name, generate_embedding(query), search_params, output_fields=["text"])
    answer = search_and_answer_question(query, results)
    
    logging.info(f"Question: {query}")
    logging.info(f"Answer: {answer}")

if __name__ == "__main__":
    main()
