from tqdm import tqdm
import yaml
import pandas as pd

import sys
sys.path.append("../rag")

from translation import PersianToEnglish

def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["translation"]
    translation_model_name = params["translation_model"]

    print("Read programs file...")
    ghazal = pd.read_parquet("data/raw/ghazal.parquet")

    print("Load translation model...")
    translation_model = PersianToEnglish(translation_model_name)

    translations = []
    for _, row_data in tqdm(ghazal.iterrows(), total=len(ghazal), desc="Translating ghazal"):
        persian_text = f"{row_data['beyt1']}\n{row_data['beyt2']}"
        translation = translation_model.translate(persian_text)
        translations.append(translation)

    ghazal['translation'] = translations
    ghazal.to_parquet("data/raw/ghazal_with_translation.parquet")

if __name__ == "__main__":
    main()    