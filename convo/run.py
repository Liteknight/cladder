import re
import time
import json
import pickle
import random
import argparse
import numpy as np
from utils import *
from generation import *
import datetime
from tqdm import tqdm
import pandas as pd

MODELS = {"qwen": "qwen3:32b", "deepseek": "deepseek-r1:32b", "magistral": "magistral"}
MODEL_NAMES = list(MODELS.keys())


def load_cladder():
    df = pd.read_json(
        "/home/finn/Documents/Python/cladder/data/cladder-v1/cladder-v1-q-hard.json"
    )
    meta = pd.read_json(
        "/home/finn/Documents/Python/cladder/data/cladder-v1-meta-models.json"
    )

    df["rung"] = df["meta"].apply(lambda x: x.get("rung"))
    df["model_id"] = df["meta"].apply(lambda x: x.get("model_id"))

    model_backgrounds = {
        m["model_id"]: m["background"] for m in meta.to_dict(orient="records")
    }
    df["background"] = df["model_id"].map(model_backgrounds)
    return df.drop(columns=["meta"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    print(timestamp)

    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--round", default=2, type=int)
    args = parser.parse_args()

    data = load_cladder()

    test_samples = data.to_dict(orient="records")[: args.num_samples]
    print(f"Number of test samples={len(test_samples)}")

    # Phase 1: Initial Response Generation for all Ollama models
    initial_results_per_model = {name: [] for name in MODEL_NAMES}

    for test_sample_idx, test_sample in tqdm(enumerate(test_samples)):
        # For each test sample, get initial predictions from all models
        for name, model_id in MODELS.items():
            result = ollama_gen_ans(
                model_id,
                test_sample,
                additional_instruc=None,
                intervene=False,
            )
            initial_results_per_model[name].append(result)
            
            # if name=="qwen":
            #     print(result)
        time.sleep(0.5)  # Small delay to avoid overwhelming Ollama server

    # Combine initial results into a single list for 'all_results'
    all_results = []
    for i in range(len(test_samples)):
        tmp = {"gold_answer": test_samples[i]["answer"]}
        for name in MODEL_NAMES:
            # Store the prediction under the key like 'qwen_output_0'
            tmp[f"{name}_output_0"] = initial_results_per_model[name][i]
        all_results.append(tmp)

    # Clean and parse initial outputs
    # Pass all model names to clean_output and parse_output
    all_results = clean_output(
        all_results, 0, model_names=MODEL_NAMES
    )
    all_results = parse_output(all_results, 0, model_names=MODEL_NAMES)
    print(
        f"Initial Round Performance: {evaluate_all(all_results, 0, model_names=MODEL_NAMES)}"
    )

    # Phase 2: Multi-Round Discussion
    for r in range(1, args.round + 1):
        print(f"----- Round {r} Discussion -----")

        # Iterate through each model for the debate round
        for current_model_name, current_model_id in MODELS.items():
            all_results = ollama_debate(
                current_model_name,
                current_model_id,
                test_samples,
                all_results,
                rounds=r,
            )

        # Clean and parse outputs after all models have debated in the current round
        all_results = clean_output(
            all_results, r, model_names=MODEL_NAMES
        )
        all_results = parse_output(all_results, r, model_names=MODEL_NAMES)
        print(
            f"Round {r} Performance: {evaluate_all(all_results, r, model_names=MODEL_NAMES)}"
        )

    # Save results
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
    output_filename = f"cladder_r{args.round}_{timestamp}.pkl"
    
    with open(output_filename, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Results saved to {output_filename}")
