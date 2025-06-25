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

random.seed(1234)


def prepare_context(
    test_sample, additional_instruc=None, intervene=False
):
    """
    Prepares the prompt context for the Ollama models.

    Args:
        test_sample (dict): The test sample containing question, background, etc.
        additional_instruc (list, optional): Additional instructions for the model. Defaults to None.
        intervene (bool, optional): Whether to include intervention in the prompt. Defaults to False.

    Returns:
        str: The fully constructed prompt string.
    """
    context = []

    # Guide for causal inference, specific to Cladder if applicable
    guide = """
You should structure your final answer as follows: Step 1) Extract the causal graph: Identify the causal graph that depicts the relationships in the scenario. The diagram should simply consist of edges denoted in "var1 -> var2" format, separated by commas.  Step 2) Determine the query type: Identify the type of query implied by the main question. Choices include "marginal probability", "conditional probability", "explaining away effect", "backdoor adjustment set", "average treatment effect", "collider bias", "normal counterfactual question", "average treatment effect on treated", "natural direct effect" or "natural indirect effect". Your answer should only be a term from the list above, enclosed in quotation marks.  Step 3) Formalize the query: Translate the query into its formal mathematical expression based on its type, utilizing the "do(·)" notation or counterfactual notations as needed.  Step 4) Gather all relevant data: Extract all the available data. Your answer should contain nothing but marginal probabilities and conditional probabilities in the form "P(...)=..." or "P(...|...)=...", each probability being separated by a semicolon. Stick to the previously mentioned denotations for the variables.  Step 5) Deduce the estimand using causal inference: Given all the information above, deduce the estimand using skills such as do-calculus, counterfactual prediction, and the basics of probabilities. Answer step by step.  Step 6) Calculate the estimand: Insert the relevant data in Step 4 into the estimand, perform basic arithmetic calculations, and derive the final answer. Step 7) Give a final yes/no answer to the question.
There is an identifiable yes/no answer, which may sometimes go against your commonsense intuition. Answer step by step, where each thinking step has AT MOST 20 words -- in other words, you need to be concise in your reasoning trace.
"""
    context.append(guide)
    context.append("\n")  # Add a newline for separation

    if "background" in test_sample and test_sample["background"]:
        context.append(f"{test_sample['background']}")
    if "given_info" in test_sample and test_sample["given_info"]:
        context.append(f"{test_sample['given_info']}")

    if intervene:
        context.append(
            "Q: "
            + test_sample["question"]
            + "\nAnswer the question given the fact that "
            + test_sample["gold_explanation"]
        )
    else:
        context.append("Q: " + test_sample["question"])

    context.append("Please answer the question with step-by-step reasoning.")
    context.append(
        "\nAlso, evaluate your confidence level (between 0.0 and 1.0) to indicate the possibility of your answer being right."
    )
    context.append(
        'Output your answer in json format, with the format as follows: {"reasoning": "", "answer": "", "confidence_level": ""}. Please strictly output in JSON format.'
    )
    context.append('Only answer yes or no in the "answer" field.')

    if additional_instruc:
        context.append("\n" + "\n".join(additional_instruc))

    context.append("Do not output irrelevant content.")
    return "\n".join(context)


# Removed prepare_context_for_chat_assistant and prepare_context_for_bard as they are not needed for Ollama


def invalid_result():
    """Returns a default invalid result dictionary."""
    result = {
        "reasoning": "",
        "answer": np.random.choice(["yes", "no"]),  # Default to yes/no for generic case
        "confidence_level": 0.0,
    }
    return result


def parse_json(model_output):
    """
    Parses a JSON string from the model's output.
    Attempts to extract a JSON block that might be embedded in other text.
    Prioritizes extraction of JSON within ```json ... ``` fences.

    Args:
        model_output (str or dict): The output from the model.

    Returns:
        dict or str: The parsed dictionary or "ERR_SYNTAX" if parsing fails.
    """
    if type(model_output) is dict:
        return model_output
    elif type(model_output) is not str:
        model_output = str(model_output)
    
    # Try to find JSON within ```json ... ``` fences first
    json_fence_match = re.search(r'```json\s*(\{.*\})\s*```', model_output, re.DOTALL)
    if json_fence_match:
        json_string = json_fence_match.group(1)
    else:
        # Fallback to finding any JSON-like object { ... }
        json_match = re.search(r'(\{.*\})', model_output, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            return "ERR_SYNTAX" # No JSON structure found

    try:
        # Attempt to clean up common LLM output issues before parsing
        # Replace single quotes with double quotes for JSON compatibility
        json_string = json_string.replace("'", '"')
        # Handle unquoted keys by adding double quotes around them
        # This regex attempts to find words that look like unquoted keys
        json_string = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', json_string)
        # Handle trailing commas that might break JSON parsing
        json_string = re.sub(r',\s*([}\]])', r'\1', json_string)

        result = json.loads(json_string)
    except json.JSONDecodeError as e:
        # If standard JSON decoding fails, try ast.literal_eval as a last resort
        try:
            result = ast.literal_eval(json_string)
            # Ensure the result is a dictionary; ast.literal_eval can return other types
            if not isinstance(result, dict):
                return "ERR_SYNTAX"
        except (SyntaxError, NameError, ValueError, TypeError):
            return "ERR_SYNTAX"
    return result


def find_idx_by_element(input_list, element):
    """Finds all indices of an element in a list."""
    return [i for i, a in enumerate(input_list) if a == element]


def find_element_by_indices(input_list, index_list):
    """Retrieves elements from a list based on a list of indices."""
    return [b for i, b in enumerate(input_list) for k in index_list if i == k]


def trans_confidence(x):
    """Transforms raw confidence score to a weighted vote."""
    x = float(x)
    if x <= 0.6:
        return 0.1
    if 0.8 > x > 0.6:
        return 0.3
    if 0.9 > x >= 0.8:
        return 0.5
    if 1 > x >= 0.9:
        return 0.8
    if x == 1:
        return 1


def parse_output(all_results, rounds, model_names):
    """
    Parses and processes the output from all models for a given round,
    calculates votes, and generates the debate prompt for the next round.

    Args:
        all_results (list): The list containing results from all agents for all rounds.
        rounds (int): The current round number.
        model_names (list): A list of descriptive names of the models (e.g., ["qwen", "deepseek", "magistral"]).

    Returns:
        list: The updated all_results list.
    """
    for i in all_results:
        certainty_vote = {}
        current_round_preds = []  # To store all predictions for the current round
        current_round_exps = []  # To store all explanations for the current round

        for name in model_names:
            output_key = f"{name}_output_{rounds}"
            if output_key in i:
                # Store prediction and explanation under new keys
                i[f"{name}_pred_{rounds}"] = i[output_key]["answer"]
                i[f"{name}_exp_{rounds}"] = (
                    f"I think the answer is {i[output_key]['answer']} because {i[output_key]['reasoning']} My confidence level is {i[output_key]['confidence_level']}."
                )

                # Add to lists for overall voting
                current_round_preds.append(i[output_key]["answer"])
                current_round_exps.append(i[f"{name}_exp_{rounds}"])

                # Update certainty vote
                if i[output_key]["answer"] not in certainty_vote:
                    certainty_vote[i[output_key]["answer"]] = (
                        trans_confidence(i[output_key]["confidence_level"]) + 1e-5
                    )
                else:
                    certainty_vote[i[output_key]["answer"]] += trans_confidence(
                        i[output_key]["confidence_level"]
                    )

        # Only proceed if there are predictions from at least one model in this round
        if current_round_preds:
            i[f"vote_{rounds}"] = current_round_preds
            i[f"exps_{rounds}"] = current_round_exps
            i[f"weighted_vote_{rounds}"] = certainty_vote

            if certainty_vote:  # Ensure certainty_vote is not empty
                i[f"weighted_max_{rounds}"] = max(
                    certainty_vote, key=certainty_vote.get
                )
            else:
                i[f"weighted_max_{rounds}"] = None  # Or some default value if no votes

            i[f"debate_prompt_{rounds}"] = ""
            vote_counts = Counter(i[f"vote_{rounds}"]).most_common(
                len(model_names)
            )  # Get counts for all answers

            # Determine majority answer
            if vote_counts:
                i[f"majority_ans_{rounds}"] = vote_counts[0][0]
            else:
                i[f"majority_ans_{rounds}"] = None  # Or some default

            # Construct debate prompt
            for v_ans, v_count in vote_counts:
                i[f"debate_prompt_{rounds}"] += (
                    f"There are {v_count} agents think the answer is {v_ans}. "
                )
                exp_index = find_idx_by_element(i[f"vote_{rounds}"], v_ans)
                group_exp = find_element_by_indices(i[f"exps_{rounds}"], exp_index)
                exp = "\n".join([f"One agent solution: {g}" for g in group_exp])
                i[f"debate_prompt_{rounds}"] += exp + "\n\n"

    return all_results


def evaluate_single_model(results, model_name):
    """Evaluates the accuracy of a single model."""
    num_correct = 0
    for i in results:
        # Check if the model's prediction exists for round 0
        if (
            f"{model_name}_output_0" in i
            and i["gold_answer"] == i[f"{model_name}_output_0"]["gold"]
        ):  # Changed from 'prediction' to output_0 and gold key
            num_correct += 1
    return num_correct / len(results)


def clean_output(all_results, rounds, model_names):
    """
    Cleans and standardizes the model outputs.

    Args:
        all_results (list): The list containing results from all agents.
        rounds (int): The current round number.
        dataset (str): The name of the dataset.
        model_names (list): A list of descriptive names of the models.

    Returns:
        list: The cleaned all_results list.
    """
    for i in all_results:
        for name in model_names:
            output_key = f"{name}_output_{rounds}"
            if output_key in i:
                original_output_dict = i[output_key]
                print(original_output_dict, output_key, i)
                raw_response_string = original_output_dict.get('response', '')
                
                # Parse the JSON from the raw response string
                parsed_json_from_response = parse_json(raw_response_string)

                # If parsing was successful and it's a dict, update the main dict's fields
                if isinstance(parsed_json_from_response, dict):
                    # Update fields from the parsed JSON, preserving 'gold' and original 'response'
                    if 'reasoning' in parsed_json_from_response:
                        original_output_dict['reasoning'] = parsed_json_from_response['reasoning']
                    if 'answer' in parsed_json_from_response:
                        original_output_dict['answer'] = parsed_json_from_response['answer']
                    if 'confidence_level' in parsed_json_from_response:
                        original_output_dict['confidence_level'] = parsed_json_from_response['confidence_level']
                else:
                    # If parsing failed, set to invalid result but try to preserve 'gold'
                    # The 'gold' key is part of the original dict from ollama_gen_ans,
                    # so we should not lose it.
                    gold_answer = original_output_dict.get('gold')
                    # Initialize with a default invalid structure
                    original_output_dict = invalid_result()
                    original_output_dict['gold'] = gold_answer # Re-add gold if available

                # Now, ensure all necessary fields are present and correctly formatted in original_output_dict
                # Ensure 'reasoning' field exists and is a string
                if 'reasoning' not in original_output_dict or original_output_dict['reasoning'] is None:
                    original_output_dict['reasoning'] = ""
                elif isinstance(original_output_dict['reasoning'], list):
                    original_output_dict['reasoning'] = " ".join(map(str, original_output_dict['reasoning']))
                elif not isinstance(original_output_dict['reasoning'], str):
                    original_output_dict['reasoning'] = str(original_output_dict['reasoning'])

                current_answer = str(original_output_dict.get('answer', '')).strip().lower()
                if current_answer not in ['yes', 'no']:
                    original_output_dict['answer'] = np.random.choice(['yes', 'no'])
                else:
                    original_output_dict['answer'] = current_answer
                        
                # Standardize 'confidence_level'
                current_confidence = original_output_dict.get('confidence_level')
                if current_confidence is None:
                    original_output_dict['confidence_level'] = 0.0
                else:
                    if isinstance(current_confidence, str) and "%" in current_confidence:
                            try:
                                original_output_dict['confidence_level'] = float(current_confidence.replace("%","")) / 100
                            except ValueError:
                                original_output_dict['confidence_level'] = 0.0
                    else:
                        try:
                            original_output_dict['confidence_level'] = float(current_confidence)
                        except (ValueError, TypeError):
                            original_output_dict['confidence_level'] = 0.0
                
                # Assign the modified dictionary back
                i[output_key] = original_output_dict
                
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

    # Evaluate aggregate predictions if they exist for the current round
    # We need to ensure these keys exist in the `all_results` before evaluating
    # For a given sample `i`, `majority_ans_X` and `weighted_max_X` will be present
    # if `parse_output` successfully processed that sample for round X.

    # Check if 'majority_ans' and 'weighted_max' fields are expected in this round
    # by checking if any sample has them from the current round
    has_majority_ans = any(
        f"majority_ans_{rounds}" in s and s[f"majority_ans_{rounds}"] is not None
        for s in all_results
    )
    has_weighted_max = any(
        f"weighted_max_{rounds}" in s and s[f"weighted_max_{rounds}"] is not None
        for s in all_results
    )

    if has_majority_ans:
        accuracies["majority_ans"] = evaluate_results(
            all_results, "majority_ans", rounds
        )
    if has_weighted_max:
        accuracies["weighted_max"] = evaluate_results(
            all_results, "weighted_max", rounds
        )

    return accuracies
