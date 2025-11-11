import faiss
import json
import numpy as np
from typing import List, Dict, Any

class DualFAISSVectorStore:
    """Searches both text and diagram FAISS indices with weighted score merging."""

    def __init__(self, text_index_path: str, diagram_index_path: str, mapping_path: str,
                 w_text: float = 0.8, w_diagram: float = 0.2):
        print("Loading dual FAISS indices...")
        # Load FAISS indices for text and diagram embeddings
        self.text_index = faiss.read_index(text_index_path)
        self.diagram_index = faiss.read_index(diagram_index_path)
        
        # Load mapping from index IDs to metadata
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.mapping = {int(k): v for k, v in json.load(f).items()}
        
        # Weights for combining text and diagram scores
        self.w_text = w_text
        self.w_diagram = w_diagram
        print(f"Loaded mapping with {len(self.mapping)} entries.")

    def search(self, query_emb: List[float], k: int = 3) -> List[Dict[str, Any]]:
        # Convert query to numpy array and normalize
        query = np.array([query_emb], dtype="float32")
        faiss.normalize_L2(query)

        # Search in both indices
        text_scores, text_idx = self.text_index.search(query, k)
        diagram_scores, diagram_idx = self.diagram_index.search(query, k)

        results = {}
        # Merge results from both indices
        for i in range(k):
            for idx, s, kind in [(text_idx[0][i], text_scores[0][i], "text"),
                                 (diagram_idx[0][i], diagram_scores[0][i], "diagram")]:
                if idx == -1 or idx not in self.mapping:  # skip invalid indices
                    continue
                if idx not in results:  # initialize result entry
                    results[idx] = {"score": 0, "text_score": 0, "diagram_score": 0, **self.mapping[idx]}
                # Assign individual scores
                if kind == "text":
                    results[idx]["text_score"] = float(s)
                else:
                    results[idx]["diagram_score"] = float(s)

        # Compute final combined score
        for r in results.values():
            # If diagram score is zero, use text score only
            if r["diagram_score"] == 0.0:
                r["score"] = r["text_score"]
            else:
                # Weighted combination of text and diagram scores
                r["score"] = self.w_text * r["text_score"] + self.w_diagram * r["diagram_score"]

        # Return top-k results sorted by combined score
        return sorted(results.values(), key=lambda x: x["score"], reverse=True)[:k]
