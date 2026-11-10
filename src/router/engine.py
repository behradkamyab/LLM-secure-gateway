import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class SemanticRouter:
    def __init__(self, faqs: List[str], threshold: float = 0.60, model_id: str = "all-MiniLM-L6-v2"):
        """
        Executes ONCE on server startup.
        Loads the model, embeds the knowledge base, and allocates the FAISS index.
        """
        self.faqs = faqs
        self.threshold = threshold
        
        # Load the Embedding Model
        self.model = SentenceTransformer(model_id)
        
        # Get the vector dimension from the model dynamically
        self.dimension = self.model.get_embedding_dimension()
        
        # create the Exact Search FAISS Index (Cosine Similarity / Inner Product)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Embed the FAQs and populate the C++ struct
        if self.faqs:
            self._build_index()

    def _build_index(self):
        # Embed the entire list of FAQs into a 2D matrix, normalized for Cosine Similarity
        faq_embeddings = self.model.encode(self.faqs, normalize_embeddings=True)
        
        # Ensure it's float32 for FAISS C++ compatibility
        faq_embeddings = np.array(faq_embeddings, dtype=np.float32)
        
        # Add to the index
        self.index.add(faq_embeddings)

    def route(self, query: str) -> Dict[str, Any]:
        """
        Executes on EVERY user request.
        Embeds a single query and searches the FAISS index.
        """
        # Embed the incoming prompt (wrapped in a list to create matrix)
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Search FAISS for the Top 1 result
        distances, indices = self.index.search(query_embedding, k=1)
        
        # Extract the math
        top_score = float(distances[0][0])
        top_index = int(indices[0][0])
        
        # The Gateway Logic
        if top_score >= self.threshold:
            return {
                "action": "CACHE_HIT",
                "score": top_score,
                "response": self.faqs[top_index]
            }
        else:
            return {
                "action": "ROUTE_TO_LLM",
                "score": top_score,
                "query_forwarded": query
            }