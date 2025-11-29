from langchain_ollama import ChatOllama
from transformers import pipeline
import torch

def load_llm(name, translation=False):
    if translation:
        return ChatOllama(model=name, temperature=0, num_predict=512)
    else:
        return ChatOllama(model=name, num_predict=512)

def load_hf_model(name, token):
    return pipeline(
        "image-text-to-text",
        model=name,
        device="cuda:1",
        torch_dtype=torch.bfloat16,
        token=token,
        tokenizer_kwargs={"use_fast": True},
        model_kwargs={
            "attn_implementation": "flash_attention_2",
        }        
    )