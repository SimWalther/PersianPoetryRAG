from __future__ import annotations
from dotenv import dotenv_values
from langchain_postgres import PGEngine
from langchain_postgres import PGVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from load_model import load_llm
from typing import Dict, List, TypedDict
from reranker import Reranker
from bm25 import BM25
from langdetect import detect
import unicodedata

# Define state for application
class State(TypedDict):
    question: str
    selected_types: list
    model_category: str
    context: List[Document]
    answer: str

class RAG(object):
    @classmethod
    async def create(
        cls,
        embedding_table_name,
        embedding_model,
        embedding_size,
        num_references,
        num_retrieved,
        reader_prompt,
        translation_prompt,
        reader_models,
        translation_model,
        ghazal_path,
        masnavi_path,
        programs_path,
        dotenv_path=".env",
    ):
        self = cls()
        # Load config
        config = dotenv_values(dotenv_path)
        pg_user = config["POSTGRES_USER"]
        pg_password = config["POSTGRES_PASSWORD"]
        pg_hostname = config["POSTGRES_HOSTNAME"]
        pg_db = config["POSTGRES_DB"]
        connection_string = f"postgresql+psycopg://{pg_user}:{pg_password}@{pg_hostname}/{pg_db}"

        self.reader_prompt_template = PromptTemplate.from_template(reader_prompt)
        self.translation_prompt_template = PromptTemplate.from_template(translation_prompt)
        self.num_references = num_references
        self.num_retrieved = num_retrieved

        # Create Postgres engine
        self.engine = PGEngine.from_connection_string(url=connection_string)

        # Load embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={
                "normalize_embeddings": True,
                "truncate_dim": embedding_size,
            },
        )        

        # Initialize LLMs
        self.reader_models = {
            model_category: load_llm(model_name)
            for model_category, model_name in reader_models.items()
        }


        self.bm25_retrievers = {
            'ghazal': BM25().create(ghazal_path, k=num_retrieved),
            'masnavi': BM25().create(masnavi_path, k=num_retrieved),
            'program': BM25().create(programs_path, k=num_retrieved)
        }

        # Create vector store
        self.vector_store = await PGVectorStore.create(
            engine=self.engine,
            table_name=embedding_table_name,
            embedding_service=self.embeddings,
            metadata_columns=["type", "number", "part", "translation"],
        )

        # Load reranker
        self.reranker = Reranker()

        # Load translation models
        self.translation_model = load_llm(translation_model, translation=True)

        # Compile application and test
        graph_builder = StateGraph(State).add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        self.rag_graph = graph_builder.compile()        

        return self
        
    def query(self, query: str, model_category: str, selected_types: list = []) -> Dict[str, object]:
        return self.rag_graph.invoke({
           "question": query,
           "model_category": model_category,
           "selected_types": selected_types
        })
    
    def similarity(self, query: str, number_results: int):    
        results = self.vector_store.similarity_search(query, k=number_results)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]

    def _retrieve(self, state: State):
        metadata_filter = dict()

        if len(state['selected_types']) >= 1:
            metadata_filter["type"] = {"$in": state['selected_types']}

        print(f"detected language: {detect(state['question'])}")
        # Translate question
        if self._is_alphabet_persian(state['question']):
            persian_question = state['question']
        else:
            translation = self.translation_prompt_template.invoke({"question": state['question']})
            persian_question = self.translation_model.invoke(translation).content
            print(f"Translated as (2): {persian_question}")
            

        retrieved_documents = []

        for doc_type, retriever in self.bm25_retrievers.items():
            if doc_type in state['selected_types']:
                retrieved_documents.extend(
                    retriever.invoke(persian_question)
                )

        retrieved_documents.extend(
            self.vector_store.similarity_search(
                f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {state['question']}",
                k=self.num_retrieved, filter=metadata_filter
            )
        )

        retrieved_documents = self.reranker.rerank(
            persian_question,
            retrieved_documents,
            self.num_references
        )

        return {
            "context": retrieved_documents,
        }

    def _generate(self, state: State):
        docs_content = ""

        for doc_rank, doc in enumerate(state["context"]):
            docs_content += f"{doc_rank}. {doc.page_content}\n"

        if state['model_category'] == 'expert':
            length = "Your answer must be detailed and in one paragraph."
        else:
            length = "Keep your answers as short as possible, think three sentences max."

        messages = self.reader_prompt_template.invoke({"question": state["question"], "context": docs_content, "length": length})
        response = self.reader_models[state['model_category']].invoke(messages).content

        return {
            "answer": response,
        }

    def _is_alphabet_persian(self, text):
        total_fa = 0

        for character in text:
            if ("ARABIC" in unicodedata.name(character)) or ("FARSI" in unicodedata.name(character)):
                total_fa += 1

        return total_fa > (len(text) - total_fa)