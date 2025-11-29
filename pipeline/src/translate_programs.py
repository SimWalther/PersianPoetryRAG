from tqdm import tqdm
import yaml
import pandas as pd
from langchain_core.prompts import PromptTemplate

import sys
sys.path.append("../rag")

from load_model import load_llm

def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["translation"]
    translation_model_name = params["translation_model"]
    translation_prompt = params["translation_prompt"]

    print("Read programs file...")
    documents = pd.read_parquet("data/raw/programs_chunked.parquet")

    print("Load translation model...")
    translation_prompt_template = PromptTemplate.from_template(translation_prompt)
    translation_model = load_llm(translation_model_name, translation=True)

    translations = []
    for _, row_data in tqdm(documents.iterrows(), total=len(documents), desc="Translating programs"):
        persian_text = row_data['program_text']
        translation = translation_prompt_template.invoke({"text": persian_text})
        english_text = translation_model.invoke(translation).content
        translations.append(english_text.strip())

    documents['translation'] = translations
    documents.to_parquet("data/raw/programs_with_translation.parquet")

if __name__ == "__main__":
    main()