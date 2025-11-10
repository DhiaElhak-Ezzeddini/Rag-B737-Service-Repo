import os
import json
import numpy as np
import faiss
import fitz  # PyMuPDF
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Tuple, List
import time
import google.generativeai as genai

# --- Configuration ---
PDF_FILE_PATH = "Boeing B737 Manual.pdf"  # Make sure this file exists
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "page_images")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "manual.index")
JSON_MAPPING_PATH = os.path.join(DATA_DIR, "manual.json")
IMAGE_DPI = 150
N_CHUNKS_PER_PAGE = 3
OVERLAP_WORDS = 10
EMBEDDING_MODEL = "text-embedding-004"  # Google's latest embedding model

# --- Helper Functions ---
def get_text_embedding(text: str, retries: int = 5) -> List[float]:
    """Generates embedding for a given text, with exponential backoff."""
    delay = 1
    for i in range(retries):
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return result['embedding']
        except Exception as e:
            if "Resource has been exhausted" in str(e) or "429" in str(e):
                print(f"Rate limit hit. Retrying in {delay}s... (Attempt {i+1}/{retries})")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Error embedding text for chunk: {e}")
                return None
    print("Failed to get embedding after all retries.")
    return None

def process_pdf_page(doc, page_num_1_based: int) -> Tuple[str, bytes]:
    """Extract text and image from one PDF page."""
    page = doc.load_page(page_num_1_based - 1)
    text = page.get_text("text").strip()
    if not text:
        text = f"[BLANK PAGE OR IMAGE-ONLY PAGE {page_num_1_based}]"
    pix = page.get_pixmap(dpi=IMAGE_DPI)
    image_bytes = pix.tobytes("png")
    return text, image_bytes

def split_text_into_chunks(text: str, n_chunks: int = 2, overlap_words: int = 50) -> List[str]:
    """Split text into roughly equal-sized chunks by word count with overlap."""
    words = text.split()
    total_words = len(words)

    if total_words <= 50:
        return [text]  # Very short page → no need to split

    base_chunk_size = total_words // n_chunks

    # Ensure overlap is not larger than chunk size
    if overlap_words >= base_chunk_size and base_chunk_size > 0:
        overlap_words = base_chunk_size // 4
    elif base_chunk_size == 0:
        overlap_words = 0

    chunks = []
    for i in range(n_chunks):
        start = i * base_chunk_size
        if i > 0:
            start -= overlap_words
        end = (i + 1) * base_chunk_size
        if i == n_chunks - 1:
            end = total_words
        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks

# --- Main Script ---
def main():
    print("Starting PDF preprocessing...")

    load_dotenv()
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)

    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: PDF file not found at {PDF_FILE_PATH}")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    try:
        doc = fitz.open(PDF_FILE_PATH)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return

    num_pages = len(doc)
    print(f"PDF loaded successfully. Found {num_pages} pages.")

    page_data_mapping = {}
    text_vectors = []
    vector_index_id = 0

    print("Processing pages (extracting text, rendering images, and generating embeddings)...")
    for page_num in tqdm(range(1, num_pages + 1), desc="Processing Pages"):
        try:
            text, image_bytes = process_pdf_page(doc, page_num)
        except Exception as e:
            print(f"Warning: Could not process page {page_num}. Skipping. Error: {e}")
            continue

        image_filename = f"page_{page_num}.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        try:
            with open(image_path, "wb") as f:
                f.write(image_bytes)
        except Exception as e:
            print(f"Warning: Could not save image for page {page_num}. Skipping. Error: {e}")
            continue

        chunks = split_text_into_chunks(text, n_chunks=N_CHUNKS_PER_PAGE, overlap_words=OVERLAP_WORDS)

        for i, chunk_text in enumerate(chunks):
            embedding = get_text_embedding(chunk_text)

            if embedding is None or len(embedding) == 0:
                print(f"Warning: Could not generate embedding for page {page_num}, chunk {i}. Skipping.")
                continue

            chunk_id_str = f"{page_num}_{i}"
            text_vectors.append(embedding)

            page_data_mapping[vector_index_id] = {
                "page_number": page_num,
                "chunk_index": i,
                "chunk_id_str": chunk_id_str,
                "text": chunk_text,
                "image_path": image_path
            }

            vector_index_id += 1

    doc.close()

    if not text_vectors:
        print("No vectors generated. Exiting.")
        return

    # Build FAISS index
    print("Creating FAISS index...")
    dimension = len(text_vectors[0])
    index = faiss.IndexFlatL2(dimension)
    vectors_np = np.array(text_vectors).astype('float32')
    index.add(vectors_np)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

    # Save metadata JSON
    with open(JSON_MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(page_data_mapping, f, indent=4)
    print(f"Page data mapping saved to {JSON_MAPPING_PATH}")

    print("\n Preprocessing complete!")
    print(f"Total chunks processed: {len(page_data_mapping)}")
    print(f"Each page was split into ~{N_CHUNKS_PER_PAGE} chunks with {OVERLAP_WORDS} word overlap.")

if __name__ == "__main__":
    main()
