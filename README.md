# qp-ai-assessment

## Project Structure

- `main.py`: Main script to orchestrate the process flow and interaction between modules.
- `modules/`: Directory containing modularized components of the project.
    - `document_loader.py`: Module for loading documents from various sources.
    - `text_processing.py`: Module for text splitting and embedding generation.
    - `milvus_operations.py`: Module for interactions with the Milvus database.
    - `chat_bot.py`: Module for the chat bot logic.
    - `config.py`: Configuration module for handling environment variables.

## Usage

1. **Load Documents**: Use the `document_loader.py` module to load documents into the system.
2. **Process Text**: Utilize the `text_processing.py` module to split text and generate embeddings.
3. **Store in Milvus**: Use the `milvus_operations.py` module to create a Milvus collection and insert documents.
4. **Interact with Chat Bot**: Call the `search_and_answer_question()` function from the `chat_bot.py` module to interact with the chat bot.

## Screenshoot of the working code



<img width="1015" alt="Screenshot 2024-02-08 at 12 11 53 AM" src="https://github.com/delirium0712/qp-ai-assessment/assets/25646098/5012f89a-b06c-4f86-988d-1b9c692aecb9">
