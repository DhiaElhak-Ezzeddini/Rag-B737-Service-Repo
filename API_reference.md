# API Reference – Multimodal Boeing 737 RAG System

## Overview
The Multimodal Boeing 737 RAG API provides a **Retrieval-Augmented Generation (RAG)** interface for textual and diagrammatic content from Boeing 737 manuals. Built with **FastAPI**, it enables structured, multimodal responses.

---

## Base URL
```bash
http://<HOST>:8000/
```

```YAML
- Port: `8000` (default)  
- `<HOST>`: server IP or `localhost`  

## Authentication
Requires a Google Gemini API key in a `.env` file:

## env
GOOGLE_GEMINI_API_KEY=your_api_key_here
```
## Endpoints
### GET /

Test endpoint to verify API status.

**Response:**
```json
{
  "message": "Welcome to the Multimodal Boeing 737 RAG API"
}
```

### POST /query

Retrieves relevant information and generates multimodal responses.

**Request Body:**

```json
{
    "question": "string",
    "top_k": 5
}
```

**Response:**

```json
{
  "answer": "string",
  "chunks": [
    {
      "text": "string",
      "diagram": "url_or_reference",
      "source": "string"
    }
  ]
}
```

**Example (cURL):**
```bash 
curl -X POST "http://localhost:8000/query" \
-H "Content-Type: application/json" \
-d '{"question": "Engine inspection procedure?", "top_k":3}'
```

**Configuration**

- **TEXT_INDEX_PATH**: FAISS index for textual data

- **DIAGRAM_INDEX_PATH**: FAISS index for diagram/tables data

- **JSON_MAPPING_PATH**: Mapping file for manual content

- **Retrieval weights**: w_text = 0.6, w_diagram = 0.4

**Usage Notes**

-   Ensure FAISS indices are pre-built.

-   Supports asynchronous queries for efficient handling.

-   Responses combine text and diagram context.

**Run Locally**:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```