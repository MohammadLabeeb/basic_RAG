# Basic RAG Implementation

This project implements a basic Retrieval-Augmented Generation (RAG) system that allows you to query your PDF documents using LLM (Large Language Model). The system processes documents, creates embeddings, and uses FAISS for efficient similarity search.

## Features

- PDF document processing and chunking
- Document embedding using Sentence Transformers
- Efficient similarity search using FAISS index
- Question answering using Llama model

## Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- PyMuPDF (fitz)
- sentence-transformers
- faiss-cpu
- transformers
- torch
- numpy

## Project Structure

- `embed_and_store.py`: Processes documents and creates embeddings
- `rag_qa.py`: Handles question-answering using the RAG approach

## Setup and Usage

1. Configure the paths in both scripts:
   ```python
   DOCUMENTS_DIR = "path/to/your/documents"  # Directory containing PDFs
   INDEX_DIR = "path/to/save/index"          # Directory to save FAISS index
   ```

2. Process your documents and create embeddings:
   ```bash
   python embed_and_store.py
   ```

3. Run the QA system:
   ```bash
   python rag_qa.py
   ```

## How It Works

1. **Document Processing (`embed_and_store.py`)**:
   - Reads PDF documents from the specified directory
   - Splits documents into overlapping chunks
   - Creates embeddings using SentenceTransformer
   - Stores embeddings in a FAISS index

2. **Question Answering (`rag_qa.py`)**:
   - Takes user questions as input
   - Retrieves relevant document chunks using similarity search
   - Uses Llama model to generate answers based on retrieved context

## Models Used

- Embedding Model: `sentence-transformers/all-minilm-l6-v2`
- Language Model: `meta-llama/Llama-3.2-1B`

## Example Usage

```bash
# First, process your documents
python embed_and_store.py

# Then start the QA system
python rag_qa.py

# Enter your questions when prompted
Enter your question: What is discussed in my documents?
```

## Customization

- Adjust chunk size and overlap in `split_into_chunks()` function
- Modify the number of retrieved chunks by changing `top_k` in `retrieve_chunks()`
- Customize generation parameters in `answer_question()`

## Limitations

- Currently only supports text in PDF documents
- Requires sufficient RAM for processing large documents
- Answer quality depends on the chunk size and retrieval accuracy

## Future Improvements

- Implement better text chunking strategies
- Add document preprocessing and cleaning
- Improve answer generation with better prompting