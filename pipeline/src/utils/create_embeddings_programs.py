import uuid
import pandas as pd
import pickle
from tqdm import tqdm
from langchain_core.documents import Document
from utils.common_embeddings import add_documents_to_vector_store

def process_program_chunk(row):
    return Document(
        page_content=row['program_text'],
        metadata={
            "id": f"P{row['program_number']}_{uuid.uuid4().hex[:4]}",
            "type": "program",
            "number": row["program_number"],
            "part": row["program_chunk"],
            "translation": row["translation"],
        },
    )

def create_program_documents(df):
    return [
        process_program_chunk(row)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating program documents")
    ]

def create_embeddings_programs(vector_store):
    print("Reading programs file...")
    programs = pd.read_parquet("data/raw/programs_with_translation.parquet")

    docs = create_program_documents(programs)
    
    print("Adding programs to vector store...")
    add_documents_to_vector_store(vector_store, docs, batch_size=25)

    with open('data/prepared/programs_documents.pkl', 'wb') as outp:
        pickle.dump(docs, outp, pickle.HIGHEST_PROTOCOL)