from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain_core.documents.base import Document

import os


def get_huggingfacehub_api_token():
    token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if token is None:
        raise ValueError("Did not find huggingfacehub_api_token. Please add an environment variable HUGGINGFACEHUB_API_TOKEN which contains it, or pass huggingfacehub_api_token as a named parameter.")
    return token

def search_and_answer_question(query, results):
    retrieved_sentences = []
    for n, hits in enumerate(results):
        for hit in hits:
            retrieved_sentences.append(Document(page_content=str(hit.entity.text)))

    llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0, "max_length": 512})
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=retrieved_sentences, question=query)
