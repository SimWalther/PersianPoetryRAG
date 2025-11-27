from langchain_ollama import ChatOllama

def load_llm(name):
    return ChatOllama(model=name)