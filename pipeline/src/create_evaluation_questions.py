from dotenv import dotenv_values
from utils.load_model import load_llm
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
import pickle
import yaml
import random

def main() -> None:
    # Load config
    config = dotenv_values(".env")

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["create_evaluation_questions"]
    num_questions = params["num_questions"]
    question_generation_model = params["question_generation_model"]
    question_generation_prompt = params["question_generation_prompt"]

    print("Load documents...")
    with open(r"data/prepared/masnavi_documents.pkl", "rb") as f:
        masnavi_documents = pickle.load(f)

    with open(r"data/prepared/ghazal_documents.pkl", "rb") as f:
        ghazal_documents = pickle.load(f)

    with open(r"data/prepared/programs_documents.pkl", "rb") as f:
        programs_documents = pickle.load(f)

    documents = masnavi_documents + ghazal_documents + programs_documents
 
    # Load question generation model
    llm = load_llm(question_generation_model)

    # Create question generation template
    question_generation_template = PromptTemplate.from_template(question_generation_prompt)

    # Generate questions
    print(f"Generating {num_questions} QA couples...")

    outputs = []
    for sampled_context in tqdm(random.sample(documents, num_questions)):

        prompt = question_generation_template.invoke({"context": sampled_context.page_content})
        generated_question_answer = llm.invoke(prompt).content

        try:
            question = generated_question_answer.split("Query: ")[-1].split("Answer: ")[0].strip()
            answer = generated_question_answer.split("Answer: ")[-1].strip()
            assert len(answer) < 300, "Answer is too long"

            outputs.append(
                {
                    "context": sampled_context.page_content,
                    "question": question,
                    "answer": answer,
                    "type": sampled_context.metadata["type"],
                    "number": sampled_context.metadata["number"],
                    "part": sampled_context.metadata["part"],
                }
            )
        except:
            print("...")
            continue

    with open('data/prepared/generated_questions.pkl', 'wb') as outp:
        pickle.dump(outputs, outp, pickle.HIGHEST_PROTOCOL)        

if __name__ == "__main__":
    main()
