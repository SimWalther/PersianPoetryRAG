from langchain_ollama import ChatOllama
from transformers import pipeline
import torch

def load_llm(name, translation=False):
    temperature = 0 if translation else 0.2
    return ChatOllama(model=name, temperature=temperature, num_predict=512)
    
def load_hf_model(name, token):
    return pipeline(
        "image-text-to-text",
        model=name,
        device="mps",
        torch_dtype=torch.bfloat16,
        token=token,
        tokenizer_kwargs={"use_fast": True},
    )