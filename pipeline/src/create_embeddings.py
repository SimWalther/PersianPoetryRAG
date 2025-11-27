from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGEngine
from langchain_postgres import Column
from langchain_postgres import PGVectorStore
from langchain_postgres.v2.indexes import IVFFlatIndex, HNSWIndex
from utils.dump_table import dump_table
from utils.create_embeddings_ghazal import create_embeddings_ghazal
from utils.create_embeddings_masnavi import create_embeddings_masnavi
from utils.create_embeddings_programs import create_embeddings_programs
from dotenv import dotenv_values
import yaml
import asyncio

async def main() -> None:
    # Load config
    config = dotenv_values(".env")

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["create_embeddings"]
    embedding_model_name = params["embedding_model"]
    embedding_table_name = params["embedding_table_name"]
    embedding_size = params["embedding_size"]

    pg_user = config["POSTGRES_USER"]
    pg_password = config["POSTGRES_PASSWORD"]
    pg_hostname = config["POSTGRES_HOSTNAME"]
    pg_db = config["POSTGRES_DB"]
    hf_token = config["HF_TOKEN"]

    connection_string = f"postgresql+psycopg://{pg_user}:{pg_password}@{pg_hostname}/{pg_db}"

    # Load embedding model
    print("Load embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        encode_kwargs={
            "normalize_embeddings": True,
            "truncate_dim": embedding_size,
        },
        model_kwargs={
            "token": hf_token
        }
    )

    # Create Postgres engine
    print("Connect to postgres...")
    engine = PGEngine.from_connection_string(url=connection_string)

    # Make sure the embedding table doesn't exists
    print(f"Recreate {embedding_table_name} table...")
    engine.drop_table(embedding_table_name)

    # Create embedding table
    await engine.ainit_vectorstore_table(
        table_name=embedding_table_name,
        vector_size=embedding_size,
        metadata_columns=[
            Column("type", "VARCHAR"),
            Column("number", "INTEGER"),
            Column("part", "INTEGER"),
            Column("translation", "VARCHAR"),
        ],
    )

    print("Create PGVectorStore...")
    # Create pgvectore store
    vector_store = await PGVectorStore.create(
        engine=engine,
        table_name=embedding_table_name,
        embedding_service=embedding_model,
        metadata_columns=["type", "number", "part", "translation"]
    )
    
    create_embeddings_ghazal(vector_store)
    create_embeddings_masnavi(vector_store)
    create_embeddings_programs(vector_store)

    # Index data
    print("Index data...")
    
    #index = HNSWIndex()
    index = IVFFlatIndex()
    
    # Make sure no index exists
    await vector_store.adrop_vector_index()

    # Create a vector index
    await vector_store.aapply_vector_index(index)

    print("Dump embeddings table...")

    # Dump embedding table
    dump_table(
        table_name=embedding_table_name,
        pg_user=pg_user,
        pg_password=pg_password,
        pg_hostname=pg_hostname,
        pg_db=pg_db,
        output_path='data/prepared/embeddings.parquet'
    )

if __name__ == "__main__":
    asyncio.run(main())