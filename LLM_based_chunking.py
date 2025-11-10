import os
import json
import numpy as np
import faiss
import fitz  # PyMuPDF
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Tuple, List, Dict, Any
import time
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# --- Configuration ---
PDF_FILE_PATH = "Boeing B737 Manual.pdf" # Make sure this file exists
DATA_DIR = "LLM_based_chunking"
IMAGE_DIR = os.path.join(DATA_DIR, "page_images")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "manual.index")
JSON_MAPPING_PATH = os.path.join(DATA_DIR, "manual.json")
IMAGE_DPI = 150
EMBEDDING_MODEL = "text-embedding-004" # Google's latest embedding model
# CHANGED: Using Pro for better reasoning and JSON adherence during chunking
GENERATION_MODEL_NAME = "gemini-2.5-pro-preview-09-2025" 

# --- Helper Functions ---

def get_text_embedding(text: str, retries: int = 5) -> List[float]:
    """Generates embedding for a given text, with exponential backoff."""
    delay = 1
    for i in range(retries):
        try:
            # Use the "RETRIEVAL_DOCUMENT" task type for preprocessing
            result = genai.embed_content(model=EMBEDDING_MODEL,
                                         content=text,
                                         task_type="RETRIEVAL_DOCUMENT")
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
    # Use "text" for structured text extraction
    text = page.get_text("text").strip()
    if not text:
        text = f"[BLANK PAGE OR IMAGE-ONLY PAGE {page_num_1_based}]"
    
    # Render the page as a high-resolution PNG
    pix = page.get_pixmap(dpi=IMAGE_DPI)
    image_bytes = pix.tobytes("png")
    return text, image_bytes

def llm_chunk_page(text: str, model: genai.GenerativeModel, retries: int = 3) -> List[str]:
    """
    Uses a Generative Model to intelligently split text into coherent RAG chunks.
    This replaces fixed-size chunking.
    """

    # Define JSON schema as a standard dictionary (new Gemini API format)
    chunk_schema = {
        "type": "object",
        "properties": {
            "chunks": {
                "type": "array",
                "description": "A list of coherent text chunks.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The extracted, coherent chunk of text."
                        }
                    },
                    "required": ["text"]
                }
            }
        },
        "required": ["chunks"]
    }

    system_instruction = (
        "You are an expert technical document editor. Your task is to split the provided text "
        "from a technical manual page into coherent, self-contained chunks suitable for Retrieval-Augmented Generation (RAG). "
        "Each chunk must represent a complete thought, procedure step, definition, or table/figure description. "
        "Do not leave out any information. Output a JSON object following the provided schema."
    )

    prompt = f"Page Text to Chunk:\n\n---\n\n{text}"
    delay = 1

    if text.startswith("[BLANK PAGE"):
        return [text]

    for i in range(retries):
        try:
            response = model.generate_content(
                contents=[prompt],
                generation_config=GenerationConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=chunk_schema,
                    temperature=0.0
                )
            )

            json_output = json.loads(response.text)
            return [item['text'] for item in json_output.get('chunks', [])]

        except Exception as e:
            if "Resource has been exhausted" in str(e) or "429" in str(e):
                print(f"Rate limit hit on chunking. Retrying in {delay}s... (Attempt {i+1}/{retries})")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Error during LLM chunking: {e}")
                return []

    print("Failed to get LLM chunks after all retries. Defaulting to single chunk.")
    return [text]



# --- Main Script ---
def main():
    print("Starting PDF preprocessing with LLM Chunking and Hybrid Retrieval preparation...")

    load_dotenv()
    # Configure the Gemini API key
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
    
    # Initialize the Generative Model for chunking once
    # NOTE: The model used here is now Pro for higher quality chunking
    llm_chunking_model = genai.GenerativeModel(GENERATION_MODEL_NAME)


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
    
    # This sequential ID is the key for FAISS and our JSON mapping
    vector_index_id = 0

    print("Processing pages (extracting text, rendering images, and generating embeddings)...")

    for page_num in tqdm(range(1, num_pages + 1), desc="Processing Pages"):
        try:
            text, image_bytes = process_pdf_page(doc, page_num)
        except Exception as e:
            print(f"Warning: Could not process page {page_num}. Skipping. Error: {e}")
            continue

        # Save image (once per page)
        image_filename = f"page_{page_num}.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        try:
            with open(image_path, "wb") as f:
                f.write(image_bytes)
        except Exception as e:
            print(f"Warning: Could not save image for page {page_num}. Skipping. Error: {e}")
            continue

        # --- LLM-based Chunking ---
        chunks = llm_chunk_page(text, llm_chunking_model)
        
        if not chunks:
            print(f"Warning: LLM returned no chunks for page {page_num}. Defaulting to full text.")
            chunks = [text] # Ensure at least the full text is processed

        for i, chunk_text in enumerate(chunks):
            
            # Use the Google API embedding model
            embedding = get_text_embedding(chunk_text)
            
            if embedding is None or len(embedding) == 0:
                print(f"Warning: Could not generate embedding for page {page_num}, chunk {i}. Skipping.")
                continue

            chunk_id_str = f"{page_num}_{i}"  # Unique identifier for de-duplication
            text_vectors.append(embedding)

            # Store metadata for both dense and keyword retrieval
            page_data_mapping[vector_index_id] = {
                "page_number": page_num,
                "chunk_index": i,
                "chunk_id_str": chunk_id_str, 
                "text": chunk_text,
                "image_path": image_path # All chunks on a page share the same image
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

    print("\n Preprocessing complete with LLM-based chunking!")
    print(f"Total chunks processed: {len(page_data_mapping)}")

if __name__ == "__main__":
    main()