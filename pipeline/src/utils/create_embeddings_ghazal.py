import uuid
import pandas as pd
import pickle
from tqdm import tqdm
from langchain_core.documents import Document
from utils.common_embeddings import add_documents_to_vector_store

def process_ghazal_beyt(row_data):
    persian_text = f"{row_data['beyt1']}\n{row_data['beyt2']}"
    return Document(
        page_content=persian_text,
        metadata={
            "id": f"G{row_data['ghazal_num']}_{uuid.uuid4().hex[:4]}",
            "type": "ghazal",
            "number": row_data["ghazal_num"],
            "part": 0,
            "translation": row_data["translation"],
        },
    )

def create_ghazal_documents(df):
    return [
        process_ghazal_beyt(row)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating ghazal documents")
    ]

def create_embeddings_ghazal(vector_store):
    print("Reading ghazal file...")
    df = pd.read_parquet("data/raw/ghazal_with_translation.parquet")
    docs = create_ghazal_documents(df)
    
    print("Adding ghazal documents to vector store...")
    add_documents_to_vector_store(vector_store, docs, batch_size=256)

    with open('data/prepared/ghazal_documents.pkl', 'wb') as outp:
        pickle.dump(docs, outp, pickle.HIGHEST_PROTOCOL)