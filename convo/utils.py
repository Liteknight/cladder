import os
import re
import ast
import json
import time

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from pydantic import BaseModel

random.seed(1234)

def invalid_result():
    """Returns a default invalid result dictionary."""
    result = {
        "reasoning": "",
        "answer": np.random.choice(["yes", "no"]),
        "confidence": 0.0,
    }
    return result


class Answer(BaseModel):
    # thinking: str
    reasoning: str
    answer: str
    confidence: float

def parse_json(model_output):
    """
    Parses a JSON string from the model's output.
    Assumes the summarizer model is producing clean JSON.

    Args:
        model_output (str or dict): The output from the model.

    Returns:
        dict or str: The parsed dictionary or "ERR_SYNTAX" if parsing fails.
    """
    if isinstance(model_output, dict):
        return model_output
    elif not isinstance(model_output, str):
        model_output = str(model_output)

    try:
        json_string = model_output.replace("'", '"')
        json_string = re.sub(r',\s*([}\]])', r'\1', json_string) # Remove trailing commas
        
        # Try to find JSON within ```json ... ``` fences first
        json_fence_match = re.search(r'```json\s*(\{.*\})\s*```', json_string, re.DOTALL)
        if json_fence_match:
            json_to_parse = json_fence_match.group(1)
        else:
            # Fallback to finding any JSON-like object { ... }
            json_match = re.search(r'(\{.*\})', json_string, re.DOTALL)
            if json_match:
                json_to_parse = json_match.group(1)
            else:
                return "ERR_SYNTAX_1" # No JSON structure found

        result = json.loads(json_to_parse)
        return result
    except json.JSONDecodeError as e:
        try:
            result = ast.literal_eval(model_output)
            if isinstance(result, dict):
                return result
            else:
                return "ERR_SYNTAX_2"
        except (SyntaxError, NameError, ValueError, TypeError):
            return "ERR_SYNTAX_3"


def parse_output(all_results, round, model_names):
    """
    Parses and processes the *structured* output from models for a given round.
    In a two-agent debate, its role for 'debate_prompt' is diminished,
    but it can still prepare individual model predictions and evaluations.

    Args:
        all_results (list): The list containing results from all agents for all rounds.
        rounds (int): The current round number.
        model_names (list): A list of descriptive names of the models (e.g., ["qwen", "deepseek"]).

    Returns:
        list: The updated all_results list.
    """
    for i in all_results:

        for name in model_names:
            structured_output_key = f"{name}_output_{round}"
            if structured_output_key in i and isinstance(i[structured_output_key], dict):
                structured_output = i[structured_output_key]

                # Store prediction and explanation under new keys
                i[f"{round}_pred_{name}"] = structured_output.get("answer")
                i[f"{round}_exp"] = [
                    f"{structured_output.get('reasoning')} Therefore, I think the final answer should be {structured_output.get('answer')}, and my confidence level is {structured_output.get('confidence')}.",
                    name,
                ]

        # i[f"debate_prompt_{rounds}"] = f"Here is your opponent's response: {i[f"{rounds}_exp_{name}"]}\n\n"

    return all_results

def evaluate_results(all_results, prefix, rounds):
    """
    Evaluates the accuracy of results for a given prefix (model or aggregate)
    and round. Handles cases where a specific round's result might be missing
    by looking at previous rounds.
    """
    num_correct = 0
    total_evaluated = 0  # Keep track of how many samples are actually evaluated
    for i in all_results:
        r_num = int(rounds)
        found_result = False
        while r_num >= 0:  # Check current round then go backwards
            eval_key = prefix + "_" + str(r_num)
            if eval_key in i:
                if i[eval_key] is not None:  # Ensure the evaluated answer is not None
                    total_evaluated += 1
                    if (
                        i["gold_answer"].strip().lower()
                        == str(i[eval_key]).strip().lower()
                    ):  # Ensure consistent comparison
                        num_correct += 1
                found_result = True
                break
            else:
                r_num -= 1
        # If no result found for any round, this sample is not evaluated for this prefix
    return num_correct / total_evaluated if total_evaluated > 0 else 0


def evaluate_all(all_results, rounds, model_names):
    """
    Evaluates the performance of all individual models and aggregate methods.

    Args:
        all_results (list): The list containing results from all agents.
        rounds (int): The current round number.
        model_names (list): A list of descriptive names of the models.

    Returns:
        dict: A dictionary of accuracies for each model and aggregate method.
    """
    accuracies = {}
    # Evaluate individual model predictions
    for name in model_names:
        accuracies[f"{name}_pred"] = evaluate_results(
            all_results, f"{name}_pred", rounds
        )

    return accuracies