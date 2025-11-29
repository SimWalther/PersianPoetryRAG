from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
import yaml
import pandas as pd

import sys
sys.path.append("../rag")

from load_model import load_llm

def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["translation"]
    translation_model_name = params["translation_model"]
    translation_prompt = params["translation_prompt"]

    print("Read programs file...")
    ghazal = pd.read_parquet("data/raw/ghazal.parquet")

    print("Load translation model...")
    translation_prompt_template = PromptTemplate.from_template(translation_prompt)
    translation_model = load_llm(translation_model_name, translation=True)

    translations = []
    for _, row_data in tqdm(ghazal.iterrows(), total=len(ghazal), desc="Translating ghazal"):
        persian_text = f"{row_data['beyt1']}\n{row_data['beyt2']}"
        translation = translation_prompt_template.invoke({"text": persian_text})
        english_text = translation_model.invoke(translation).content
        translations.append(english_text.strip())

    ghazal['translation'] = translations
    ghazal.to_parquet("data/raw/ghazal_with_translation.parquet")

if __name__ == "__main__":
    main()