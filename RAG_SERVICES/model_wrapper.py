import os
import mimetypes
import base64
import time
from typing import List, Dict, Any,Tuple
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",device="cpu")

# --- Configuration ---

# Use a multimodal model that can understand text and images
# gemini-2.5-flash-preview-09-2025 is a great choice for speed and multimodality
GENERATION_MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

# Use the latest embedding model
EMBEDDING_MODEL_NAME = "text-embedding-004"

# --- Internal Helper Functions ---

def _load_image_as_base64(image_path: str) -> Tuple[str, str]:
    """Loads an image file, base64-encodes it, and determines its MIME type."""
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found at {image_path}")
        return None, None
        
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "image/png"  # Default
        return encoded_string, mime_type
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

def _create_multimodal_prompt(question: str, chunks: List[Dict[str, Any]]) -> List[Any]:
    """
    Constructs the complex multimodal prompt for the Gemini API.
    This includes system instructions, the query, and all retrieved text/images.
    """
    
    # 1. System Instruction
    # This guides the model on its role and how to use the context.
    page_numbers = sorted(list(set([c['page_number'] for c in chunks])))
    system_instruction = f"""
    You are an expert aviation assistant for a Boeing 737.
    Your task is to answer the user's question based *only* on the provided context.
    The context consists of text and full-page images from the technical manual.
    The page numbers for the provided context are: {page_numbers}.
    
    CRITICAL RULES:
    1.  Base your answer *exclusively* on the text and images provided.
    2.  Do not add any information not present in the context.
    3.  If the answer is not in the context, state that you cannot find the information in the provided documents.
    4.  Analyze both the text and the images (diagrams, tables, schemas) to formulate your answer.
    5.  Provide a clear, concise, and accurate answer to the user's question.
    6.  Do NOT repeat the page numbers in your final answer. The system will cite them. Just provide the answer.
    """
    
    # 2. Construct the prompt parts list
    prompt_parts = [system_instruction, f"USER QUESTION: \"{question}\"\n\n--- PROVIDED CONTEXT ---"]
    
    # 3. Add all retrieved chunks (text and images)
    # We sort by page number to give the model a logical flow.
    sorted_chunks = sorted(chunks, key=lambda x: x['page_number'])
    
    for chunk in sorted_chunks:
        page_num = chunk['page_number']
        text = chunk['text']
        image_path = chunk['image_path']
        print(text)
        # Add the text context
        prompt_parts.append(f"\n--- START: Context from Page {page_num} (Text) ---")
        prompt_parts.append(text)
        prompt_parts.append(f"--- END: Context from Page {page_num} (Text) ---")
        
        # Add the image context
        base64_data, mime_type = _load_image_as_base64(image_path)
        if base64_data:
            prompt_parts.append(f"\n--- START: Context from Page {page_num} (Image) ---")
            prompt_parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64_data
                }
            })
            prompt_parts.append(f"--- END: Context from Page {page_num} (Image) ---")
        else:
            prompt_parts.append(f"[Image for page {page_num} could not be loaded]")
            
    return prompt_parts


# --- Public API Functions ---

async def get_text_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT", retries: int = 5) -> List[float]:
    """
    Generates an embedding for a given text.
    
    Args:
        text: The string to embed.
        task_type: "RETRIEVAL_DOCUMENT" for preprocessing, "RETRIEVAL_QUERY" for queries.
        retries: Number of retries for API calls.
    
    Returns:
        The embedding vector (list of floats).
    """
    delay = 1
    for i in range(retries):
        try:
            """result = await genai.embed_content_async(
                model=EMBEDDING_MODEL_NAME,
                content=text,
                task_type=task_type
            )
            return result['embedding']"""
            embedding = embedding_model.encode(text)
            return embedding
        except Exception as e:
            if "Resource has been exhausted" in str(e) or "429" in str(e):
                print(f"Rate limit hit on embedding. Retrying in {delay}s... (Attempt {i+1}/{retries})")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Error embedding text: {e}")
                return None
    print("Failed to get embedding after all retries.")
    return None

async def get_multimodal_rag_response(question: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Generates the final answer using the multimodal RAG chain.
    
    Args:
        question: The user's question.
        chunks: The list of retrieved page-chunks (from FAISSVectorStore).
    
    Returns:
        The generated text answer.
    """
    try:
        # 1. Initialize the generative model
        model = genai.GenerativeModel(GENERATION_MODEL_NAME)
        
        # 2. Construct the prompt
        prompt_parts = _create_multimodal_prompt(question, chunks)
        
        # 3. Configure generation
        # We can use JSON output to be more robust, but text is fine for this task.
        generation_config = GenerationConfig(
            temperature=0.1,  # Low temp for factual, grounded answers
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
        )

        # 4. Call the API asynchronously
        response = await model.generate_content_async(
            prompt_parts,
            generation_config=generation_config
        )
        
        # 5. Return the text part of the response
        return response.text
        
    except Exception as e:
        print(f"Error during multimodal generation: {e}")
        return f"Error: Could not generate a response. {str(e)}"