import os
import json
import fitz  # PyMuPDF
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Paths to required resources
# TODO: Replace these paths with your actual paths
INDEX_DIR = r""  # Directory where the FAISS index is stored
EMBEDDINGS_FILE = "doc_embeddings.json"  # Path to the JSON file containing document embeddings

# Load the Llama model and tokenizer
print("Loading model and tokenizer...")
# if model is loaded from a local directory, specify the path
# MODEL_PATH = r""  # Path to the downloaded Llama model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')

# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load embedding model
# if model is loaded from a local directory, specify the path
# embedding_model = SentenceTransformer(r"")

embedding_model = SentenceTransformer('sentence-transformers/all-minilm-l6-v2')

def read_document(file_path):
    """Read text from a PDF or other supported document."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_index():
    """Load the FAISS index and metadata."""
    index = faiss.read_index(os.path.join(INDEX_DIR, "document_index.faiss"))
    with open(os.path.join(INDEX_DIR, EMBEDDINGS_FILE), "r") as f:
        metadata = json.load(f)
    return index, metadata["metadata"]

def retrieve_chunks(query, index, metadata, top_k=3):
    """Retrieve the most relevant document chunks based on the query with improved filtering."""
    # Encode query
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    
    # Get more candidates initially
    k_candidates = min(top_k * 2, len(metadata))
    distances, indices = index.search(np.array(query_embedding), k_candidates)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            chunk_meta = metadata[idx]
            # Only include chunks with similarity score above threshold
            if distances[0][i] < 1.2:  # Adjust threshold as needed
                results.append((chunk_meta, distances[0][i]))
    
    # Sort by distance and take top_k
    results.sort(key=lambda x: x[1])
    return results[:top_k]

def answer_question(query, retrieved_chunks):
    """Answer the query using the Llama model with the context from the retrieved chunks."""
    # Combine text of retrieved chunks with clear separation
    context = "\n---\n".join([chunk_meta["chunk"] for chunk_meta, _ in retrieved_chunks])[:2048]

    input_text = (
        f"You are a helpful AI assistant. Answer the following question based on the provided context. "
        f"Be specific, accurate, and concise. If you cannot find the answer in the context, say 'I cannot find enough information to answer this question.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Improved generation parameters
    output = model.generate(
        input_ids["input_ids"],
        attention_mask=input_ids["attention_mask"],
        max_new_tokens=200,
        min_length=10,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=3,                 # Using beam search for better quality
        do_sample=True,              # Enable sampling
        temperature=0.7,             # Add temperature for more natural responses
        top_p=0.9,                   # Add top_p for better text quality
        repetition_penalty=1.2       # Prevent repetition
    )

    # Decode only the new tokens
    input_length = input_ids["input_ids"].shape[1]
    new_tokens = output[0][input_length:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Clean up the answer
    answer = answer.strip()
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    if "Context:" in answer:
        answer = answer.split("Context:")[0].strip()

    return answer

if __name__ == "__main__":
    # Load FAISS index and document metadata
    print("Loading index and metadata...")
    index, document_names = load_index()
    print("\nRAG QA System Ready!")
    print('Enter your questions. Type "exit", "quit", or press Ctrl+C to end the session.')
    
    try:
        while True:
            # Get user input
            query = input("\nEnter your question: ").strip()
            
            # Check for exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nThank you for using the RAG QA system. Goodbye!")
                break
            
            # Skip empty questions
            if not query:
                print("Please enter a valid question.")
                continue
                
            print("Retrieving relevant documents...")
            retrieved_docs = retrieve_chunks(query, index, document_names, top_k=3)

            print("\nRetrieved Documents:")
            for doc, score in retrieved_docs:
                print(f"- File: {doc['file_name']}")
                print(f"  Similarity score: {score:.4f}")
                print(f"  Chunk: {doc['chunk'][:150]}...")  # Show first 150 chars of chunk

            print("\nGenerating answer...")
            answer = answer_question(query, retrieved_docs)
            print(f"\nAnswer: {answer}")
            
            print("\n" + "-"*50)  # Add separator between QA pairs
            
    except KeyboardInterrupt:
        print("\n\nSession terminated by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Session terminated due to error.")
    finally:
        pass
