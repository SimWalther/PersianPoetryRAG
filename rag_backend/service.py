from __future__ import annotations
import bentoml
import yaml
from typing import Dict
from bentoml.exceptions import InvalidArgument

import sys
sys.path.append("../rag")

from rag import RAG

@bentoml.service(
    traffic={"timeout": 600},
    http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["*"],
            "access_control_allow_methods": ["*"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
        },
    }    
)
class RAGService:
    def __init__(self, config="params.yaml"):
        params = yaml.safe_load(open(config))
        rag_params = params["rag"]

        self.reader_prompt = rag_params["reader_prompt"]
        self.translation_prompt = rag_params["translation_prompt"]
        self.embedding_table_name = rag_params["embedding_table_name"]
        self.embedding_model = rag_params["embedding_model"]
        self.embedding_size= rag_params["embedding_size"]
        self.num_references = rag_params["num_references"]
        self.num_retrieved = rag_params["num_retrieved"]
        self.translation_model = rag_params["translation_model"]
        self.reader_models = {
            "fast": rag_params["reader_models_fast"],
            "expert": rag_params["reader_models_expert"],
        }
        self.dotenv_path=".env"
    
    @bentoml.on_startup
    async def create_rag(self):
        self.rag_service = await RAG.create(
            embedding_table_name=self.embedding_table_name,
            embedding_model=self.embedding_model,
            embedding_size=self.embedding_size,
            num_references=self.num_references,
            num_retrieved=self.num_retrieved,
            reader_prompt=self.reader_prompt,
            translation_prompt=self.translation_prompt,
            reader_models=self.reader_models,
            translation_model=self.translation_model,
            ghazal_path='data/ghazal_documents.pkl',
            masnavi_path='data/masnavi_documents.pkl',
            programs_path='data/programs_documents.pkl',
            dotenv_path=self.dotenv_path
        )
    
    @bentoml.api
    def rag(self, query: str, model_category: str, selected_types: list) -> Dict[str, object]:
        if model_category not in self.reader_models.keys():
            print(f"Model category {model_category} does not exists.")
            return {}
        
        response = self.rag_service.query(query, model_category, selected_types)

        # Return the response, while stripping everything unnecessary.
        return {
            'answer': response['answer'],
            'references': response['context']
        }