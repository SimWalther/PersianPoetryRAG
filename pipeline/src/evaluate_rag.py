import pandas as pd
import yaml
import matplotlib.pyplot as plt
import json
import numpy as np
from dotenv import dotenv_values
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
from utils.load_model import load_llm
import asyncio

import sys
sys.path.append("../rag")

from rag import RAG

async def main() -> None:
    # Load config
    config = dotenv_values(".env")

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))
    evaluate_params = params["evaluate_rag"]
    embedding_params = params["create_embeddings"]
    rag_params = params["rag"]

    evaluation_prompt = PromptTemplate.from_template(evaluate_params["evaluation_prompt"])

    # Load evaluation dataset
    print("Read evaluation dataset...")
    evaluation_dataset = pd.read_parquet("data/prepared/evaluation_dataset.parquet")
    
    # Create RAG
    print("Create RAG...")

    rag = await RAG.create(
        embedding_table_name = embedding_params["embedding_table_name"],
        embedding_model = embedding_params["embedding_model"],
        embedding_size = embedding_params["embedding_size"],
        num_references = rag_params["num_references"],
        num_retrieved = rag_params["num_retrieved"],
        reader_prompt = rag_params["reader_prompt"],
        translation_prompt = rag_params["translation_prompt"],
        reader_models = {'fast': rag_params["reader_model"]},
        translation_model = rag_params["translation_model"],
        ghazal_path = 'data/prepared/ghazal_documents.pkl',
        masnavi_path = 'data/prepared/masnavi_documents.pkl',
        programs_path = 'data/prepared/programs_documents.pkl',
        dotenv_path = ".env"
    )

    # Gather RAG answers
    print("Gather RAG answers...")
    outputs = []

    for _, example in tqdm(evaluation_dataset.iterrows(), total=len(evaluation_dataset)):
        question = example["question"]
        answer = rag.query(query=question, model_category='fast', selected_types=['ghazal', 'program', 'masnavi'])['answer'].model_dump()

        generated_answer = answer['answer'] if 'answer' in answer else ''
        retrieved_docs = answer['references'] if 'references' in answer else []

        result = {
            "question": question,
            "true_answer": example["answer"],
            "type": example["type"],
            "number": example["number"], 
            "part": example["part"], 
            "generated_answer": generated_answer,
            "retrieved_docs": retrieved_docs,
        }
        
        outputs.append(result)

    # Load judge model
    print("Load judge model...")
    judge_model = load_llm(evaluate_params["judge_model"])

    print("Judge RAG answers...")
    # Judge RAG answers
    for experiment in tqdm(outputs, total=len(outputs)):
        eval_result = judge_model.invoke(
            evaluation_prompt.invoke({"instruction": experiment["question"], "response": experiment["generated_answer"], "reference_answer": experiment["true_answer"]})
        ).content
        
        feedback, score = [
            item.strip() for item in eval_result.split("[RESULT]")
        ]
        experiment[f"eval_score"] = score
        experiment[f"eval_feedback"] = feedback    
    
    # Get scores
    print("Save metrics...")
    scores = [int(output['eval_score']) for output in outputs]
    mean_judge_scores = np.mean(scores)

    with open("evaluation/metrics.json", "w") as f:
        json.dump({"mean_judge_scores": mean_judge_scores}, f)

    plt.figure(figsize=(3, 3))
    plt.title("Evaluation score distribution")
    plt.boxplot(scores)
    plt.xticks([])
    plt.savefig("evaluation/plots/judge_scores.png")

    with open('evaluation/rag_results.json', "w") as f:
        json.dump(outputs, f)    

if __name__ == "__main__":
    asyncio.run(main())
