# Requires transformers>=4.51.0
import torch
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name='jinaai/jina-reranker-v2-base-multilingual'):
        self.model = CrossEncoder(
            model_name,
            max_length=1024,
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            trust_remote_code=True
        )
    
    def rerank(self, query: str, documents: list, top_k: int):
        if len(documents) <= top_k:
            return documents
        
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Add scores to documents and sort
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:top_k]]