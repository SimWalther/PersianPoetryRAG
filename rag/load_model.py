from langchain_ollama import ChatOllama
from transformers import pipeline
import torch

def load_llm(name, translation=False):
    if translation:
        return ChatOllama(model=name, temperature=0, num_predict=512)
    else:
        return ChatOllama(model=name, num_predict=512)