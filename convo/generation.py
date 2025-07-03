import os
import time
from tqdm import tqdm
import requests
from utils import prepare_context, parse_json, invalid_result
from pydantic import BaseModel

OLLAMA_URL = "http://localhost:11434/api/generate"

class Answer(BaseModel):
    thinking: str
    reasoning: str
    answer: str
    confidence: float

def ollama_gen_ans(model, sample, additional_instruc=None, intervene=False):
    
    prompt = prepare_context(sample, additional_instruc=additional_instruc, intervene=intervene)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.6,
            "top_p": 0.95
        },
        "format": Answer.model_json_schema()
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    
    content = response.json()["response"]

    return {
        "response": content.strip(),
        "gold": sample["answer"].strip().lower()
    }

def ollama_debate(model_name, model_id, test_samples, all_results, rounds):
    """
    Orchestrates a debate round for a specific Ollama model.

    Args:
        model_name (str): The descriptive name of the model (e.g., 'qwen').
        model_id (str): The actual Ollama model ID (e.g., 'qwen3:32b').
        test_samples (list): List of test samples.
        all_results (list): The list containing results from all agents for all rounds.
        rounds (int): The current round number.

    Returns:
        list: The updated all_results list after the debate round for this model.
    """
    r = '_' + str(rounds-1) # Suffix for previous round's debate prompt
    # Suffix for current round's output
    current_output_key = f'{model_name}_output_{rounds}'

    for i, s in tqdm(enumerate(all_results)):
        # Check if the current model has already produced an output for this round
        # and if there's a debate prompt from the previous round
        if current_output_key not in s and f'debate_prompt_{rounds-1}' in s and len(s[f'debate_prompt_{rounds-1}']):
            additional_instruc = [
                "\n\nCarefully review the following solutions from other agents. If you identify any flaws in their reasoning, you should point them out and suggest corrections.",
                "Clearly state which point(s) of view you agree or disagree with and why. You may defend your previous answer, or be persuaded to another solution, as you see fit.\n\n",
                s[f'debate_prompt_{rounds-1}'], # Use the debate prompt generated in the previous round
                # "It is exceedingly important that you output your answer in json format, with the format as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format."
            ]
            
            result = ollama_gen_ans(
                model_id, # Pass the actual Ollama model ID
                test_samples[i],
                additional_instruc=additional_instruc,
                intervene=False,
            )
            s[current_output_key] = result
    return all_results
