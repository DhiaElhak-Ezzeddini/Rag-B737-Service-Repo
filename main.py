import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio

from RAG_SERVICES.vector_store import DualFAISSVectorStore
from RAG_SERVICES.model_wrapper import get_text_embedding, get_multimodal_rag_response

# --- Configuration ---
TEXT_INDEX_PATH = os.getenv("TEXT_INDEX_PATH","new_data/manual_text.index")
DIAGRAM_INDEX_PATH = os.getenv("DIAGRAM_INDEX_PATH","new_data/manual_diagram.index")
JSON_MAPPING_PATH = os.getenv("JSON_MAPPING_PATH","new_data/manual.json")

# --- FastAPI Setup ---
app = FastAPI(title="Multimodal Boeing 737 RAG API", description="Retrieval-Augmented Generation with Text + Diagram Context.")

# --- Load API Key ---
load_dotenv()
api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GOOGLE_GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=api_key)

# --- Load Vector Store ---
vector_store = DualFAISSVectorStore(TEXT_INDEX_PATH, DIAGRAM_INDEX_PATH, JSON_MAPPING_PATH, w_text=0.6, w_diagram=0.4)

# --- Request Model ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

# --- API Routes ---
@app.get("/")
def root():
    return {"message": "Welcome to the Multimodal Boeing 737 RAG API"}

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        # 1. Embed the query
        query_embedding = await get_text_embedding(request.question, task_type="RETRIEVAL_QUERY")
        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate embedding for query.")

        # 2. Search both indices
        retrieved_chunks = vector_store.search(query_embedding, k=request.top_k)
        if not retrieved_chunks:
            return {"answer": "No relevant information found in the manual.", "chunks": []}

        # 3. Generate final multimodal answer
        final_answer = await get_multimodal_rag_response(request.question, retrieved_chunks)

        return final_answer

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Entry Point ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
