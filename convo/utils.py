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


def prepare_context(
    test_sample, additional_instruc=None, intervene=False, force_json=False
):
    """
    Prepares the prompt context for the Ollama models.

    Args:
        test_sample (dict): The test sample containing question, background, etc.
        additional_instruc (list, optional): Additional instructions for the model. Defaults to None.
        intervene (bool, optional): Whether to include intervention in the prompt. Defaults to False.
        force_json (bool): Whether to include the JSON format instruction in the prompt.

    Returns:
        str: The fully constructed prompt string.
    """
    context = []
    guide = """
You will be asked a causal reasoning question. You should structure your final answer as follows: Step 1) Extract the causal graph: Identify the causal graph that depicts the relationships in the scenario. The diagram should simply consist of edges denoted in "var1 -> var2" format, separated by commas.  Step 2) Determine the query type: Identify the type of query implied by the main question. Choices include "marginal probability", "conditional probability", "explaining away effect", "backdoor adjustment set", "average treatment effect", "collider bias", "normal counterfactual question", "average treatment effect on treated", "natural direct effect" or "natural indirect effect". Your answer should only be a term from the list above, enclosed in quotation marks.  Step 3) Formalize the query: Translate the query into its formal mathematical expression based on its type, utilizing the "do(·)" notation or counterfactual notations as needed.  Step 4) Gather all relevant data: Extract all the available data. Your answer should contain nothing but marginal probabilities and conditional probabilities in the form "P(...)=..." or "P(...|...)=...", each probability being separated by a semicolon. Stick to the previously mentioned denotations for the variables.  Step 5) Deduce the estimand using causal inference: Given all the information above, deduce the estimand using skills such as do-calculus, counterfactual prediction, and the basics of probabilities. Answer step by step.  Step 6) Calculate the estimand: Insert the relevant data in Step 4 into the estimand, perform basic arithmetic calculations, and derive the final answer. Step 7) Give a final yes/no answer to the question.
There is an identifiable yes/no answer, which may sometimes go against your commonsense intuition. Answer step by step, where each thinking step has AT MOST 20 words -- in other words, you need to be concise in your reasoning trace.
Be confident in your thinking: while answers may be unintuitive, there are no trick questions, and answers will be obvious once calculated.
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

    context.append(
        "\nAlso, evaluate your confidence level (between 0.0 and 1.0) to indicate the possibility of your answer being right."
    )

    if force_json:
        context.append(
            'Output with JSON format as follows: {"reasoning": "", "answer": "", "confidence": ""}. Your internal thoughts go in the first, and your final, polished reasoning goes in the second.'
        )
        context.append('Only answer yes or no in the "answer" field.')

    if additional_instruc:
        context.append("\n" + "\n".join(additional_instruc))

    context.append("Do not output irrelevant content.")
    return "\n".join(context)


def prepare_summarizer_prompt(raw_text):
    """
    Prepares a prompt for the summarizer model to extract JSON.
    """
    prompt = f"""
    The following is a response from another AI model. Your task is to extract the 'reasoning', 'answer' (which should be 'yes' or 'no'), and 'confidence' (a float between 0.0 and 1.0) from the text below. If any field is missing or unclear, use an empty string for reasoning, 'yes' or 'no' (randomly chosen) for answer, and 0.0 for confidence.

    Response:
    ---
    {raw_text}
    ---

    Output with JSON format as follows: {{"reasoning": "", "answer": "", "confidence": ""}}. Only answer yes or no in the "answer" field.
    """
    return prompt


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
            return "ERR_SYNTAX_1" # No JSON structure found

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
                return "ERR_SYNTAX_2"
        except (SyntaxError, NameError, ValueError, TypeError):
            return "ERR_SYNTAX_3"
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
    This function now expects the `_output_` field to contain the raw model response,
    and the structured JSON will be derived by `clean_output`.

    Args:
        all_results (list): The list containing results from all agents for all rounds.
        rounds (int): The current round number.
        model_names (list): A list of descriptive names of the models (e.g., ["qwen", "deepseek"]).

    Returns:
        list: The updated all_results list.
    """
    for i in all_results:
        certainty_vote = {}
        current_round_preds = []  # To store all predictions for the current round
        current_round_exps = []  # To store all explanations for the current round

        for name in model_names:
            output_key = f"{name}_output_{rounds}"
            if output_key in i and isinstance(i[output_key], dict):
                # Store prediction and explanation under new keys
                i[f"{name}_pred_{rounds}"] = i[output_key]["answer"]
                i[f"{name}_exp_{rounds}"] = (
                    f"I think the answer is {i[output_key]['answer']} because {i[output_key]['reasoning']} My confidence level is {i[output_key]['confidence']}."
                )

                # Add to lists for overall voting
                current_round_preds.append(i[output_key]["answer"])
                current_round_exps.append(i[f"{name}_exp_{rounds}"])

                # Update certainty vote
                if i[output_key]["answer"] not in certainty_vote:
                    certainty_vote[i[output_key]["answer"]] = (
                        trans_confidence(i[output_key]["confidence"]) + 1e-5
                    )
                else:
                    certainty_vote[i[output_key]["answer"]] += trans_confidence(
                        i[output_key]["confidence"]
                    )

        # Only proceed if there are predictions from at least one model in this round
        if current_round_preds:
            i[f"vote_{rounds}"] = current_round_preds
            i[f"exps_{rounds}"] = current_round_exps
            i[f"weighted_vote_{rounds}"] = certainty_vote

            if certainty_vote:
                i[f"weighted_max_{rounds}"] = max(
                    certainty_vote, key=certainty_vote.get
                )
            else:
                i[f"weighted_max_{rounds}"] = None 

            i[f"debate_prompt_{rounds}"] = ""
            vote_counts = Counter(i[f"vote_{rounds}"]).most_common(
                len(model_names)
            ) 

            if vote_counts:
                i[f"majority_ans_{rounds}"] = vote_counts[0][0]
            else:
                i[f"majority_ans_{rounds}"] = None

            # Construct debate prompt from the raw responses, not just the parsed explanations
            for v_ans, v_count in vote_counts:
                i[f"debate_prompt_{rounds}"] += (
                    f"There are {v_count} agents that think the answer is {v_ans}. "
                )
                exp_index = find_idx_by_element(i[f"vote_{rounds}"], v_ans)
                group_exp = find_element_by_indices(i[f"exps_{rounds}"], exp_index)
                exp = "\n".join([f"Here is one agent's solution: {g}" for g in group_exp])
                i[f"debate_prompt_{rounds}"] += exp + "\n\n"

    return all_results


def clean_output(all_results, rounds, model_names, summarizer_model_id):
    """
    Cleans and standardizes the model outputs by using a summarizer model
    to parse raw responses into a structured JSON format.

    Args:
        all_results (list): The list containing results from all agents.
        rounds (int): The current round number.
        model_names (list): A list of descriptive names of the models.
        summarizer_model_id (str): The Ollama model ID for the summarizer.

    Returns:
        list: The cleaned all_results list.
    """
    for i in all_results:
        for name in model_names:
            raw_output_key = f"{name}_raw_output_{rounds}"
            structured_output_key = f"{name}_output_{rounds}"

            if raw_output_key in i:
                raw_response_string = i[raw_output_key].get('response', '')
                gold_answer = i[raw_output_key].get('gold') 

                # Use the summarizer model to parse the raw response
                # We need a dummy sample for prepare_context, but the actual content is in raw_response_string
                # The summarizer_model_id will be used to call ollama_gen_ans
                from generation import ollama_summarize_output # Import locally to avoid circular dependency

                parsed_structured_output = ollama_summarize_output(
                    summarizer_model_id,
                    raw_response_string,
                    gold_answer # Pass gold answer to summarizer for consistency
                )
                
                # If parsing was successful and it's a dict, update the main dict's fields
                if isinstance(parsed_structured_output, dict):
                    # Ensure all necessary fields are present and correctly formatted
                    final_structured_output = {}
                    final_structured_output['reasoning'] = parsed_structured_output.get('reasoning', '')
                    final_structured_output['answer'] = str(parsed_structured_output.get('answer', '')).strip().lower()
                    if final_structured_output['answer'] not in ['yes', 'no']:
                        final_structured_output['answer'] = np.random.choice(['yes', 'no'])
                    
                    current_confidence = parsed_structured_output.get('confidence')
                    if current_confidence is None:
                        final_structured_output['confidence'] = 0.0
                    else:
                        if isinstance(current_confidence, str) and "%" in current_confidence:
                                try:
                                    final_structured_output['confidence'] = float(current_confidence.replace("%","")) / 100
                                except ValueError:
                                    final_structured_output['confidence'] = 0.0
                        else:
                            try:
                                final_structured_output['confidence'] = float(current_confidence)
                            except (ValueError, TypeError):
                                final_structured_output['confidence'] = 0.0
                    
                    final_structured_output['gold'] = gold_answer
                    i[structured_output_key] = final_structured_output
                else:
                    # If summarization failed, set to invalid result and preserve 'gold'
                    invalid = invalid_result()
                    invalid['gold'] = gold_answer
                    i[structured_output_key] = invalid
                
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

