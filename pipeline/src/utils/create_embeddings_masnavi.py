import uuid
import pandas as pd
import pickle
from tqdm import tqdm
from langchain_core.documents import Document
from utils.common_embeddings import add_documents_to_vector_store

def process_masnavi_beyt(row_data, row_bakhsh):
    persian_text = f"{row_data['beyt1']}\n{row_data['beyt2']}"
    translation_text = f"{row_data['beyt1_en']}\n{row_data['beyt2_en']}"
    return Document(
        page_content=persian_text,
        metadata={
            "id": f"M{row_data['book']}_{row_bakhsh['bakhsh']}_{uuid.uuid4().hex[:4]}",
            "type": "masnavi",
            "number": row_data["book"],
            "part": row_bakhsh["bakhsh"],
            "translation": translation_text,
        },
    )

def create_masnavi_documents(masnavi, masnavi_bakhsh):
    masnavi_docs = []
    for _, row_bakhsh in tqdm(masnavi_bakhsh.iterrows(), total=len(masnavi_bakhsh), desc="Creating masnavi documents"):
        for _, row_data in masnavi.query(
            'book == @row_bakhsh.book and number >= @row_bakhsh.first_beyt and number <= @row_bakhsh.last_beyt'
        ).iterrows():
            masnavi_docs.append(process_masnavi_beyt(row_data, row_bakhsh))
    return masnavi_docs

def create_embeddings_masnavi(vector_store):
    print("Reading masnavi files...")
    masnavi = pd.read_parquet("data/raw/masnavi.parquet")
    masnavi_bakhsh = pd.read_parquet("data/raw/masnavi_bakhsh.parquet")

    docs = create_masnavi_documents(masnavi, masnavi_bakhsh)
    
    print("Adding masnavi documents to vector store...")
    add_documents_to_vector_store(vector_store, docs, batch_size=256)

    with open('data/prepared/masnavi_documents.pkl', 'wb') as outp:
        pickle.dump(docs, outp, pickle.HIGHEST_PROTOCOL)