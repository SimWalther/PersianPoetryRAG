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
    
    def rerank(self, query: str, documents: list, score_threshold = 0.15):
        # if len(documents) <= top_k:
        #     return documents
        
        seen_docs = set()
        unique_documents = []

        for doc in documents:
            if doc.page_content not in seen_docs:
                seen_docs.add(doc.page_content)
                unique_documents.append(doc)

        pairs = [[query, doc.page_content] for doc in unique_documents]
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Add scores to documents and sort
        scored_docs = list(zip(unique_documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Filter doc by score
        filtered = [doc for doc, score in scored_docs if score > score_threshold]
        
        return filtered