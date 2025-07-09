import numpy as np
import os
import time
from tqdm import tqdm
import requests
from utils import parse_json, invalid_result
from pydantic import BaseModel

OLLAMA_URL = "http://localhost:11434/api/generate"

class Answer(BaseModel):
    # thinking: str
    reasoning: str
    answer: str
    confidence: float

def ollama_gen_ans(model, sample, additional_instruc=None, response_format=None):
    """
    Generates an answer from an Ollama model.

    Args:
        model (str): The Ollama model ID.
        sample (dict): The test sample containing the question and other context.
        additional_instruc (list, optional): Additional instructions for the model. Defaults to None.
        response_format (dict, optional): JSON schema for the desired output format. If None, no specific format is requested.

    Returns:
        dict: A dictionary containing the model's raw response and the gold answer.
    """
    # Prepare the context for the thinking models (qwen, deepseek)
    if response_format:
        prompt = prepare_summarizer_prompt(sample)
    else:
        prompt = prepare_context(
            sample,
            additional_instruc=additional_instruc,
        )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.6,
            "top_p": 0.95
        }
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

    try:
        summarizer_response = ollama_gen_ans(
            summarizer_model_id,
            dummy_sample,
            response_format=Answer.model_json_schema(),
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


def ollama_debate(current_model_name, current_model_id, test_samples, all_results, rounds, opponent_model_name, summarizer_model_id):
    """
    Orchestrates a debate round for a specific Ollama model, responding to the opponent's previous turn.
    This function now immediately summarizes the raw output into structured JSON.

    Args:
        current_model_name (str): The descriptive name of the current model (e.g., 'qwen').
        current_model_id (str): The actual Ollama model ID for the current model.
        test_samples (list): List of test samples.
        all_results (list): The list containing results from all agents for all rounds.
        rounds (int): The current round number.
        opponent_model_name (str): The descriptive name of the opponent model in the debate.
        summarizer_model_id (str): The Ollama model ID for the summarizer.

    Returns:
        list: The updated all_results list after the debate round for this model.
    """
    # Key for the opponent's structured output from the previous turn
    # This assumes the opponent has already generated and had their output summarized in the previous round or turn.
    opponent_previous_output_key = f'{opponent_model_name}_output_{rounds-1}'

    # Keys for the current model's raw and structured output for the current round
    current_raw_output_key = f'{current_model_name}_raw_output_{rounds}'
    current_structured_output_key = f'{current_model_name}_output_{rounds}'

    for i, s in tqdm(enumerate(all_results)):
        # Check if the current model hasn't produced output for this round yet
        # AND if the opponent has a valid structured output from the previous round to debate against.
        if current_structured_output_key not in s and opponent_previous_output_key in s and s[opponent_previous_output_key] is not None:
            opponent_structured_output = s[opponent_previous_output_key]

            # Extract details from opponent's structured output
            opponent_exp = opponent_structured_output.get('reasoning', 'No reasoning provided.')
            opponent_answer = opponent_structured_output.get('answer', 'unknown')
            opponent_confidence = opponent_structured_output.get('confidence', 0.0)

            debate_instruc = [
                f"\n\nYour opponent, {opponent_model_name}, has provided the following solution:",
                f"Answer: {opponent_answer}",
                f"Reasoning: {opponent_exp}",
                f"Confidence: {opponent_confidence}",
                "\nCarefully review your opponent's solution. If you identify any flaws in their reasoning, you should point them out and suggest corrections.",
                # "You may defend your previous answer (if applicable), or be persuaded to another solution, as you see fit.\n\n",
            ]

            # Generate the raw response from the current thinking model
            result = ollama_gen_ans(
                current_model_id,
                test_samples[i],
                additional_instruc=debate_instruc,
                response_format=None,
            )
            s[current_raw_output_key] = result 
            s[current_structured_output_key] = ollama_summarize_output(
                summarizer_model_id,
                result.get('response', ''),
                result.get('gold')
            )
        elif current_structured_output_key not in s: # If opponent_previous_output_key is missing or invalid
            s[current_raw_output_key] = {"response": "No opponent output to debate.", "gold": test_samples[i]["answer"].strip().lower()}
            s[current_structured_output_key] = invalid_result() # Mark as invalid due to lack of debate context

    return all_results


def prepare_context(
    test_sample, additional_instruc=None,
):
    """
    Prepares the prompt context for the Ollama models.

    Args:
        test_sample (dict): The test sample containing question, background, etc.
        additional_instruc (list, optional): Additional instructions for the model. Defaults to None.
    Returns:
        str: The fully constructed prompt string.
    """
    context = []
    
    if not additional_instruc:
        context.append("""
    You will be asked a causal reasoning question. You should structure your final answer as follows: Step 1) Extract the causal graph: Identify the causal graph that depicts the relationships in the scenario. The diagram should simply consist of edges denoted in "var1 -> var2" format, separated by commas.  Step 2) Determine the query type: Identify the type of query implied by the main question. Choices include "marginal probability", "conditional probability", "explaining away effect", "backdoor adjustment set", "average treatment effect", "collider bias", "normal counterfactual question", "average treatment effect on treated", "natural direct effect" or "natural indirect effect". Your answer should only be a term from the list above, enclosed in quotation marks.  Step 3) Formalize the query: Translate the query into its formal mathematical expression based on its type, utilizing the "do(·)" notation or counterfactual notations as needed.  Step 4) Gather all relevant data: Extract all the available data. Your answer should contain nothing but marginal probabilities and conditional probabilities in the form "P(...)=..." or "P(...|...)=...", each probability being separated by a semicolon. Stick to the previously mentioned denotations for the variables.  Step 5) Deduce the estimand using causal inference: Given all the information above, deduce the estimand using skills such as do-calculus, counterfactual prediction, and the basics of probabilities. Answer step by step.  Step 6) Calculate the estimand: Insert the relevant data in Step 4 into the estimand, perform basic arithmetic calculations, and derive the final answer. Step 7) Give a final yes/no answer to the question.
    There is an identifiable yes/no answer, which may sometimes go against your commonsense intuition. Be confident in your thinking: while answers may be unintuitive, there are no trick questions, and answers will be obvious once calculated.\n
    """)

    if "background" in test_sample and test_sample["background"]:
        context.append(f"{test_sample['background']}")
    if "given_info" in test_sample and test_sample["given_info"]:
        context.append(f"{test_sample['given_info']}")

    context.append(
        "\nYou should explicitly state your level of confidence in your answer (between 0.0 and 1.0)."
    )
    if additional_instruc:
        context.append("\n" + "\n".join(additional_instruc))

    return "\n".join(context)


def prepare_summarizer_prompt(raw_text):

    prompt = f"""
    The following is a response from another AI model. Your task is to extract the 'reasoning', 'answer' (which should be 'yes' or 'no'), and 'confidence' (a float between 0.0 and 1.0) from the text below. If confidence is not specified, assume it is 1.0.
    You should maintain the same first- or third-person perspective used by the original writer.

    Response:
    ---
    {raw_text}
    ---

    Output with JSON format as follows: {{"reasoning": "", "answer": "", "confidence": ""}}. Only answer yes or no in the "answer" field.
    """
    return prompt
