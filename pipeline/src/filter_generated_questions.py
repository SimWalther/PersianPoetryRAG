from dotenv import dotenv_values
from utils.load_model import load_llm
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
import pickle
import yaml
import pandas as pd
import matplotlib.pyplot as plt

def main() -> None:
    # Load config
    config = dotenv_values(".env")

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["filter_generated_questions"]
    critique_model_name = params["critique_model"]
    critique_prompt = PromptTemplate.from_template(params["critique_prompt"])
    score_threshold = params["score_threshold"]

    print("Load generated questions...")
    with open(r"data/prepared/generated_questions.pkl", "rb") as f:
        generated_questions = pickle.load(f)

    print("Load critique model...")
    # Load critique model
    llm = load_llm(critique_model_name)

    print("Generating critique for each QA couple...")

    for output in tqdm(generated_questions):
        evaluation = llm.invoke(
            critique_prompt.invoke({"context": output["context"], "question": output["question"]})
        ).content

        try:
            output.update(
                {
                    f"score": int(evaluation.split("Rating: ")[-1].strip()),
                    f"eval": evaluation.split("Rating: ")[-2].split("Evaluation: ")[1],
                }
            )
        except Exception as e:
            print(e)
            continue

    generated_questions = pd.DataFrame.from_dict(generated_questions)

    print("Evaluation dataset before filtering:")

    print(
        generated_questions[
            [
                "question",
                "answer",
                "score",
            ]
        ]
    )

    plt.title("Generated questions score distribution")
    plt.boxplot([
        generated_questions['score'],
    ])
    plt.savefig('evaluation/plots/questions_score_distribution.png')

    generated_questions_filtered = generated_questions.query("score >= @score_threshold")

    print("============================================")
    print("Final evaluation dataset:")
    print(
        generated_questions_filtered[
            [
                "question",
                "answer",
                "score",
            ]
        ]
    )

    generated_questions_filtered.to_parquet("data/prepared/evaluation_dataset.parquet")    

if __name__ == "__main__":
    main()
