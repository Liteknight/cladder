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

random.seed()
MODELS = {"qwen": "qwen3:32b", "deepseek": "deepseek-r1:32b"}
MODEL_NAMES = list(MODELS.keys())

# TODO: experiment with lighter-weight model
SUMMARIZER_ID = "qwen3:32b"

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
    parser.add_argument("--responses", default=4, type=int) # Total responses for agents: 2 will mean each agent speaks once
    args = parser.parse_args()

    data = load_cladder()

    test_samples = data.to_dict(orient="records")[: args.num_samples]
    print(f"Number of test samples={len(test_samples)}")

    all_results = []
    for i in range(len(test_samples)):
        all_results.append({"gold_answer": test_samples[i]["answer"], "initial_question": test_samples[i]["question"]})

    # Randomly select the first debater
    first_name = random.choice(MODEL_NAMES)
    second_name = [name for name in MODEL_NAMES if name != first_name][0]
    print(f"Initial speaker: {first_name}")
    print(f"Second speaker: {second_name}")

    # Phase 1: Initial Response Generation for the first debater
    print(f"----- Round 0: Initial Response by {first_name} -----")
    for i, s in tqdm(enumerate(all_results)):
        current_sample = test_samples[i]
        
        prompt = prepare_context(current_sample)
        
        # First debater generates initial raw response
        raw_result = ollama_gen_ans(
            MODELS[first_name],
            prompt,
            response_format=None,
        )
        s[f"{first_name}_raw_output_0"] = raw_result

        s[f"{first_name}_output_0"] = ollama_summarize_output(
            SUMMARIZER_ID,
            raw_result
        )
    
    # Parse output for evaluation purposes for Round 0 (only first debater's output)
    all_results = parse_output(all_results, 0, model_names=[first_name])
    print(f"Initial Round Performance: {evaluate_all(all_results, 0, model_names=[first_name])}")


    # Phase 2: Multi-Round Debate
    speaker_order = {}

    for r_idx in range(1, args.responses):
        if r_idx % 2 != 0: # Odd turns are second_debater's response
            speaker_order[r_idx] = second_name
        else: # Even turns are first_debater's response
            speaker_order[r_idx] = first_name

    for r in range(1, args.responses): # r represents the current debate turn/round number
        
        curr_name = speaker_order[r]
        curr_id = MODELS[curr_name]
        opp_name = first_name if curr_name == second_name else second_name
        
        print(f"----- Round {r}: {curr_name} debates {opp_name}'s output from Round {r-1} -----")

        all_results = ollama_debate(
            curr_name,
            curr_id,
            test_samples, # test_samples are needed for the base question/context
            all_results,
            rounds=r, # Current round/turn number for storing results
            opp_name=opp_name,
            summ_id=SUMMARIZER_ID
        )
        
        # Parse output for evaluation after each debate turn for the speaker who just spoke
        all_results = parse_output(all_results, r, model_names=[curr_name])
        print(
            f"Round {r} Performance (after {curr_name}'s turn): {evaluate_all(all_results, r, model_names=[first_name, second_name])}"
        )

    # Save results
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
    out_file = f"r{args.responses}_{timestamp}.pkl"

    with open(out_file, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Results saved to {out_file}")