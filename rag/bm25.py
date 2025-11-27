from langchain_community.retrievers import BM25Retriever
from shekar import Normalizer, Stemmer, WordTokenizer
import pickle

# Module-level instances (created once)
_normalizer = Normalizer()
_stemmer = Stemmer()
_wordTokenizer = WordTokenizer()

def preprocess_persian(text):
    text = _normalizer(text)
    tokens = _wordTokenizer(text)
    tokens = [_stemmer(token) for token in tokens]
    return tokens

class BM25:
    def create(self, documents_path, k):
        index_path = documents_path.replace('.pkl', '_bm25_retriever.pkl')
        
        try:
            with open(index_path, 'rb') as f:
                print(f"Loading pre-computed BM25 from {index_path}...")
                return pickle.load(f)
        except FileNotFoundError:
            with open(documents_path, 'rb') as f:
                documents = pickle.load(f)
            
            print(f"Creating BM25 retriever for {documents_path}...")
            retriever = BM25Retriever.from_documents(
                documents, k=k, preprocess_func=preprocess_persian
            )
            
            # Save the entire retriever
            with open(index_path, 'wb') as f:
                pickle.dump(retriever, f)
            
            return retriever