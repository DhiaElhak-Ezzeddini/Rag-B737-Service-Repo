import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import time
from typing import List
from RAG_SERVICES.vector_store import FAISSVectorStore
from RAG_SERVICES.model_wrapper import get_multimodal_rag_response, get_text_embedding

# --- App Initialization ---
# Load environment variables from .env file
load_dotenv()

# Gemini API key
api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GOOGLE_GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=api_key)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal RAG API",
    description="API for querying a technical PDF manual using multimodal RAG.",
    version="1.0.0"
)

# --- Global Variables ---
vector_store: FAISSVectorStore = None

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5  # pages to retrieve

class QueryResponse(BaseModel):
    answer: str
    pages: List[int]

# --- API Events ---
@app.on_event("startup")
def startup_event():
    global vector_store

    faiss_path = os.getenv("FAISS_PATH", "data/manual.index")
    mapping_path = os.getenv("MAPPING_PATH", "data/manual.json")

    if not os.path.exists(faiss_path) or not os.path.exists(mapping_path):
        print("Error: FAISS index or JSON mapping not found.")
        print("Please run `python preprocessing.py` first.")
        vector_store = None
    else:
        try:
            vector_store = FAISSVectorStore(faiss_path, mapping_path)
            print("FAISS index and JSON mapping loaded successfully.")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            vector_store = None

# --- API Endpoints ---
@app.get("/", summary="Health Check")
def read_root():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Multimodal RAG API is running."}

@app.post("/query", response_model=QueryResponse, summary="Query the Manual")
async def query_api(request: QueryRequest):
    """
    1. Embeds the user's question
    2. Retrieves the top_k relevant pages (text + image)
    3. Sends all context to Gemini to generate an answer
    4. Returns the answer and the cited page numbers
    """
    if vector_store is None:
        raise HTTPException(
            status_code=500,
            detail="Vector store not initialized. Did you run preprocessing.py?"
        )

    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start_time = time.time()

    try:
        # 1. Embed the user's question
        print(f"Embedding query: \"{request.question}\"")
        query_vector = await get_text_embedding(
            request.question,
            task_type="RETRIEVAL_QUERY"
        )

        # 2. Retrieve relevant pages
        print(f"Searching for top {request.top_k} relevant pages...")
        retrieved_chunks = vector_store.search(query_vector, k=request.top_k)

        page_numbers = sorted(list(set([chunk['page_number'] for chunk in retrieved_chunks])))
        print(f"Retrieved context from pages: {page_numbers}")

        # 3. Generate a response using the multimodal model
        print("Generating answer with multimodal context...")
        generated_answer = await get_multimodal_rag_response(
            question=request.question,
            chunks=retrieved_chunks
        )

        end_time = time.time()
        print(f"Query processed in {end_time - start_time:.2f} seconds.")

        # 4. Return the response
        return QueryResponse(
            answer=generated_answer,
            pages=page_numbers
        )

    except Exception as e:
        print(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- Run the Server ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
