import numpy as np
import os
import time
from tqdm import tqdm
import requests
from utils import prepare_context, parse_json, invalid_result, prepare_summarizer_prompt
from pydantic import BaseModel

OLLAMA_URL = "http://localhost:11434/api/generate"

class Answer(BaseModel):
    # thinking: str
    reasoning: str
    answer: str
    confidence: float

def ollama_gen_ans(model, sample, additional_instruc=None, intervene=False, response_format=None, force_json=False):
    """
    Generates an answer from an Ollama model.

    Args:
        model (str): The Ollama model ID.
        sample (dict): The test sample containing the question and other context.
        additional_instruc (list, optional): Additional instructions for the model. Defaults to None.
        intervene (bool, optional): Whether to include intervention in the prompt. Defaults to False.
        response_format (dict, optional): JSON schema for the desired output format. If None, no specific format is requested.
        force_json (bool): Whether to explicitly include JSON format instruction in the prompt text.

    Returns:
        dict: A dictionary containing the model's raw response and the gold answer.
    """
    # Prepare the context for the thinking models (qwen, deepseek)
    # The JSON format instruction is now controlled by force_json
    prompt = prepare_context(
        sample,
        additional_instruc=additional_instruc,
        intervene=intervene,
        force_json=force_json
    )
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.6,
            "top_p": 0.95
        },
    }
    
    # Add format to payload only if response_format is provided
    if response_format:
        payload["format"] = response_format

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    
    content = response.json()["response"]

    return {
        "response": content.strip(),
        "gold": sample["answer"].strip().lower()
    }

def ollama_summarize_output(summarizer_model_id, raw_text_to_summarize, original_gold_answer):
    """
    Uses a dedicated summarizer model to parse raw, unstructured output into a structured JSON.

    Args:
        summarizer_model_id (str): The Ollama model ID for the summarizer.
        raw_text_to_summarize (str): The raw text output from a thinking model.
        original_gold_answer (str): The gold answer associated with the original sample.

    Returns:
        dict: The parsed structured output (reasoning, answer, confidence) or an invalid result.
    """
    # Create a dummy sample for prepare_context, as the actual content is in raw_text_to_summarize
    # The 'question' field is used to pass the raw text to the summarizer prompt.
    dummy_sample = {"question": raw_text_to_summarize, "answer": original_gold_answer}

    summarizer_prompt_text = prepare_summarizer_prompt(raw_text_to_summarize)

    try:
        summarizer_response = ollama_gen_ans(
            summarizer_model_id,
            dummy_sample, # Use dummy_sample for context
            additional_instruc=[summarizer_prompt_text], # Pass the specific summarizer prompt
            response_format=Answer.model_json_schema(),
            force_json=True
        )
        # The 'response' field of summarizer_response will contain the JSON string
        parsed_json = parse_json(summarizer_response.get('response', ''))
        
        if isinstance(parsed_json, dict):
            # Ensure confidence is a float
            try:
                parsed_json['confidence'] = float(parsed_json.get('confidence', 0.0))
            except (ValueError, TypeError):
                parsed_json['confidence'] = 0.0
            
            # Ensure answer is 'yes' or 'no'
            answer = str(parsed_json.get('answer', '')).strip().lower()
            if answer not in ['yes', 'no']:
                parsed_json['answer'] = np.random.choice(['yes', 'no'])
            else:
                parsed_json['answer'] = answer

            return parsed_json
        else:
            print(f"Summarizer failed to parse JSON: {parsed_json}")
            return invalid_result()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return invalid_result()


def ollama_debate(model_name, model_id, test_samples, all_results, rounds, other_model_name, summarizer_model_id):
    """
    Orchestrates a debate round for a specific Ollama model, alternating turns.

    Args:
        model_name (str): The descriptive name of the current model (e.g., 'qwen').
        model_id (str): The actual Ollama model ID for the current model.
        test_samples (list): List of test samples.
        all_results (list): The list containing results from all agents for all rounds.
        rounds (int): The current round number.
        other_model_name (str): The descriptive name of the other model in the debate.
        summarizer_model_id (str): The Ollama model ID for the summarizer.

    Returns:
        list: The updated all_results list after the debate round for this model.
    """
    # Suffix for current round's raw output and previous round's structured output
    current_raw_output_key = f'{model_name}_raw_output_{rounds}'
    previous_structured_output_key = f'{other_model_name}_output_{rounds-1}' # Output of the opponent from previous turn

    for i, s in tqdm(enumerate(all_results)):
        # Check if the current model has already produced an output for this round
        # and if there's a previous debate turn from the opponent
        if current_raw_output_key not in s and previous_structured_output_key in s:
            
            # Construct the debate prompt based on the opponent's last structured output
            opponent_exp = s[previous_structured_output_key].get('reasoning', 'No reasoning provided.')
            opponent_answer = s[previous_structured_output_key].get('answer', 'unknown')
            opponent_confidence = s[previous_structured_output_key].get('confidence', 0.0)

            debate_instruc = [
                f"\n\nYour opponent, {other_model_name}, has provided the following solution:",
                f"Answer: {opponent_answer}",
                f"Reasoning: {opponent_exp}",
                f"Confidence: {opponent_confidence}",
                "\nCarefully review your opponent's solution. If you identify any flaws in their reasoning, you should point them out and suggest corrections.",
                "Clearly state which point(s) of view you agree or disagree with and why. You may defend your previous answer, or be persuaded to another solution, as you see fit.\n\n",
                "Provide your full reasoning and final answer. Do NOT output in JSON format. Your response will be summarized later."
            ]
            
            # Generate the raw response from the current thinking model
            result = ollama_gen_ans(
                model_id, # Pass the actual Ollama model ID
                test_samples[i],
                additional_instruc=debate_instruc,
                intervene=False,
                response_format=None, 
                force_json=False 
            )
            s[current_raw_output_key] = result # Store the raw output

            # Summarize the raw output immediately after generation
            # The structured output will be stored under f"{model_name}_output_{rounds}"
            s[f"{model_name}_output_{rounds}"] = ollama_summarize_output(
                summarizer_model_id,
                result.get('response', ''),
                result.get('gold')
            )
            
    return all_results

