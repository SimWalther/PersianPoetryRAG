from tqdm import tqdm

def add_documents_to_vector_store(vector_store, documents, batch_size=100):
    print("Add them to PGVectorStore...")
    # Add all documents to pgvector store
    # Note: we do it by batch to avoid postgres max parameters issues
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(batch)