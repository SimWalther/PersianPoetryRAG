from tqdm import tqdm
from shekar import Normalizer
import yaml
import pandas as pd
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

norm = Normalizer()

def clean_text(text):
    text = norm(text)
    text = re.sub(r'\n+', '.', text)
    text = re.sub(r'\.{2,}', '.', text)
    return text

def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["chunking"]
    chunk_size = params["chunk_size"]
    chunk_overlap = params["chunk_overlap"]
    
    print("Read programs file...")
    programs = pd.read_parquet("data/raw/programs.parquet")

    print("Clean program texts")
    programs['program_text'] = programs['program_text'].apply(clean_text)

    # Define chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = []
    for _, row_data in tqdm(programs.iterrows(), desc="Chunking programs"):
        for program_chunk, chunk in enumerate(text_splitter.split_text(row_data['program_text'])):
            chunks.append({
                'program_text': chunk,
                'program_number': row_data['program_number'],
                'program_chunk': program_chunk,
            })

    chunks_df = pd.DataFrame(chunks)
    chunks_df.to_parquet("data/raw/programs_chunked.parquet")

if __name__ == "__main__":
    main()