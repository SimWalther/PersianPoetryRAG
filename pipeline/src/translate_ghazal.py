from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
import yaml
import pandas as pd
import sys
from datasets import Dataset
sys.path.append("../rag")
from load_model import load_hf_model
from dotenv import dotenv_values

def format_prompt(instruction, context):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": context}
            ]
        }
    ]

def main() -> None:
    config = dotenv_values(".env")
    hf_token = config["HF_TOKEN"]
    params = yaml.safe_load(open("params.yaml"))["translation"]
    translation_model_name = params["translation_model"]
    translation_prompt = params["translation_prompt"]
    batch_size = params["batch_size"]
    
    print("Read programs file...")
    ghazal = pd.read_parquet("data/raw/ghazal.parquet")
    
    print("Load translation model...")
    translation_model = load_hf_model(translation_model_name, token=hf_token)
    
    # Prepare texts for batch processing
    print("Preparing texts for batch translation...")
    persian_texts = [
        f"{row['beyt1']}\n{row['beyt2']}" 
        for _, row in ghazal.iterrows()
    ]
    
    # Create prompts for all texts
    prompts = [
        format_prompt(translation_prompt, f"{text}\nTranslation:")
        for text in persian_texts
    ]
    
    # Create dataset for efficient batching
    dataset = Dataset.from_dict({"prompts": prompts})
    
    print(f"Translating {len(ghazal)} ghazals...")
    
    # Process in batches
    translations = []
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(dataset), batch_size), total=num_batches, desc="Translating ghazal"):
        batch_prompts = dataset[i:i+batch_size]["prompts"]
        current_batch_size = len(batch_prompts)
        
        # Generate translations for batch
        batch_results = translation_model(
            text=batch_prompts,
            max_new_tokens=512,
            temperature=0,
            batch_size=current_batch_size,
        )
        
        # Extract translations from results
        for result in batch_results:
            english_text = result[0]["generated_text"][-1]["content"]
            translations.append(english_text.strip())
    
    # Verify we got all translations
    assert len(translations) == len(ghazal), f"Expected {len(ghazal)} translations, got {len(translations)}"
    
    # Add translations to dataframe
    ghazal['translation'] = translations
    ghazal.to_parquet("data/raw/ghazal_with_translation.parquet")
    
    print(f"Translated {len(translations)} ghazals")

if __name__ == "__main__":
    main()