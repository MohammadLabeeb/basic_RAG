import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

# Specify the directory containing your documents and where to save the embeddings
DOCUMENTS_DIR = r"C:\Users\I012127\Desktop\RAG\RAG_Docs_test"  # Path to directory containing your PDF/Word files
INDEX_DIR = r"C:\Users\I012127\Desktop\RAG\RAG_Docs_test_vector_store"  # Directory to save the FAISS index
EMBEDDINGS_FILE = "doc_embeddings.json"

# Load the SentenceTransformer model
embedding_model = SentenceTransformer(r"C:\Users\I012127\Desktop\RAG\models\all-minilm-l6-v2")

def read_document(file_path):
    """Read text from a PDF or other supported document."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_into_chunks(text, chunk_size=300, overlap_size=50):
    """
    Split text into chunks with overlapping segments.
    
    Args:
        text (str): The text to split.
        chunk_size (int): Number of words per chunk.
        overlap_size (int): Number of overlapping words between chunks.
    
    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap_size  # Slide window with overlap

        # Ensure start does not go below 0
        if start < 0:
            start = 0

    return chunks

def embed_and_store():
    """Embed documents and store their embeddings."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    document_chunks = []
    metadata = []

    # Read and chunk documents
    for file_name in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, file_name)
        if file_name.endswith(".pdf"):  # Extendable to .docx or other formats
            print(f"Processing {file_name}...")
            text = read_document(file_path)
            for chunk in split_into_chunks(text):
                document_chunks.append(chunk)
                metadata.append({"file_name": file_name, "chunk": chunk})

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = embedding_model.encode(document_chunks, convert_to_tensor=False)

    # Create and save FAISS index
    print("Saving embeddings to FAISS index...")
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(np.array(embeddings))

    faiss.write_index(index, os.path.join(INDEX_DIR, "document_index.faiss"))

    # Save metadata for reference
    with open(os.path.join(INDEX_DIR, EMBEDDINGS_FILE), "w") as f:
        json.dump({"metadata": metadata}, f)

    print("Embedding and storing completed!")

if __name__ == "__main__":
    embed_and_store()