import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from gateway.common.schema import RouteResponse

class SemanticRouter:
    def __init__(self, faqs: dict[str, str], threshold: float, model_id: str):
        self.threshold = threshold
        self.faq_texts = list(faqs.keys())
        self.faq_labels = list(faqs.values())
        
        # Native PyTorch init (No ONNX)
        self.model = SentenceTransformer(model_id) 
        self.dimension = self.model.get_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        
        if self.faq_texts:
            self._build_index()

    def _build_index(self) -> None:
        embeddings = self.model.encode(self.faq_texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)

    def route(self, query: str) -> RouteResponse:
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        distances, indices = self.index.search(query_embedding, k=1)
        top_score = float(distances[0][0])
        top_index = int(indices[0][0])
        
        if top_score >= self.threshold:
            return RouteResponse(
                destination=self.faq_labels[top_index],
                confidence_score=top_score
            )
            
        return RouteResponse(
            destination="LLM_PIPELINE",
            confidence_score=top_score
        )