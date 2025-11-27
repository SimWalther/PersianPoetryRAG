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
    programs = pd.read_parquet("data/raw/programs_chunked.parquet")

    print("Load translation model...")
    translation_model = PersianToEnglish(translation_model_name)

    translations = []
    for _, row_data in tqdm(programs.iterrows(), total=len(programs), desc="Translating programs"):
        persian_text = row_data['program_text']
        translation = translation_model.translate(persian_text)
        translations.append(translation)

    programs['translation'] = translations
    programs.to_parquet("data/raw/programs_with_translation.parquet")

if __name__ == "__main__":
    main()