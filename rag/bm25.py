from rank_bm25 import BM25Okapi
from shekar import Normalizer, Lemmatizer, WordTokenizer
from shekar.preprocessing import (
  PunctuationRemover,
  NonPersianRemover,
  DiacriticRemover,
)
import pickle
import numpy as np

# Module-level instances (created once)
_normalizer = Normalizer()
_lemmatizer = Lemmatizer()
_wordTokenizer = WordTokenizer()
_clean_text = NonPersianRemover() | DiacriticRemover() | PunctuationRemover()

def preprocess_persian(text):
    text = _clean_text(text)
    text = _normalizer(text)
    tokens = _wordTokenizer(text)
    tokens = [_lemmatizer(token) for token in list(tokens)]
    return tokens

class BM25:
    def __init__(self):
        self.documents = []

    def create(self, documents_paths, index_name, k1, b):
        index_path = f'data/{index_name}_bm25_retriever.pkl'
        
        try:
            with open(index_path, 'rb') as f:
                print(f"Loading pre-computed BM25 from {index_path}...")
                self.retriever = pickle.load(f)

            for path in documents_paths:
                with open(path, 'rb') as f:
                    self.documents.extend(pickle.load(f))

        except FileNotFoundError:
            for path in documents_paths:
                with open(path, 'rb') as f:
                    self.documents.extend(pickle.load(f))
            
            print(f"Creating BM25 retriever for {index_name}...")

            preprocessed_documents = [
                preprocess_persian(str(doc.page_content))
                for doc in self.documents
            ]
                
            self.retriever = BM25Okapi(preprocessed_documents, b=b, k1=k1)

            # Save the entire retriever
            with open(index_path, 'wb') as f:
                pickle.dump(self.retriever, f)

            print(f"Saved bm25 retriever: {index_path}")

        return self

    def retrieve(self, query: list[str], limit: int, threshold: float):
        scores = self.retriever.get_scores(query)
        sorted_scores_indices = np.argsort(scores)[::-1][:limit]
        sorted_scores = np.asarray(scores)[sorted_scores_indices]
        
        high_scores = sorted_scores > threshold

        print(sorted_scores[high_scores])
        return np.asarray(self.documents)[sorted_scores_indices][high_scores]
