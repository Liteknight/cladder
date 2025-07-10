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


def ollama_gen_ans(model, prompt, response_format=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.6, "top_p": 0.95},
    }

    # Add format to payload only if response_format is provided
    if response_format:
        payload["format"] = response_format
        payload["think"] = False

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    content = response.json()["response"]

    # return {
    #     "response": content.strip(),
    #     "gold": sample["answer"].strip().lower()
    # }
    return content.strip()


def summarize(summ_id, raw_text):
    prompt = f"""
    The following is a response from another AI model. Your task is to extract the 'reasoning', 'answer' (which should be 'yes' or 'no'), and 'confidence' (a float between 0.0 and 1.0) from the text below. If confidence is not specified, assume it is 1.0.
    You should maintain the same first- or third-person perspective used by the original writer.

    Response:
    ---
    {raw_text}
    ---

    Output with JSON format as follows: {{"reasoning": "", "answer": "", "confidence": ""}}. Only answer yes or no in the "answer" field.
    """

    print("Summarizing...")
    summarizer_response = ollama_gen_ans(
        summ_id,
        prompt,
        response_format=Answer.model_json_schema(),
    )
    # The 'response' field of summarizer_response will contain the JSON string
    parsed_json = parse_json(summarizer_response)

    print("Done summary.")

    if isinstance(parsed_json, dict):
        # Ensure confidence is a float
        try:
            parsed_json["confidence"] = float(parsed_json.get("confidence", 0.0))
        except (ValueError, TypeError):
            parsed_json["confidence"] = 0.0

        # Ensure answer is 'yes' or 'no'
        answer = str(parsed_json.get("answer", "")).strip().lower()
        if answer not in ["yes", "no"]:
            parsed_json["answer"] = np.random.choice(["yes", "no"])
        else:
            parsed_json["answer"] = answer

        return parsed_json
    else:
        print(f"Summarizer failed to parse JSON: {parsed_json}")
        return invalid_result()
    # except Exception as e:
    #     print(f"Error during summarization: {e}")
    #     return invalid_result()


def ollama_debate(
    curr_name,
    curr_id,
    test_samples,
    all_results,
    round,
    opp_name,
    summ_id,
):

    for i, s in tqdm(enumerate(all_results)):
        history = []

        for j in range(0, round):
            key = s[f"{j}_exp"]

            if key[1] == opp_name:
                history.append("Here's what your opponent said:\n")
            else:
                history.append(
                    "Here's what you replied:\n"
                    if j > 0
                    else "Here's your original answer:\n"
                )
            history.append(key[0])

        prompt = prepare_prompt(test_samples[i], round, history)

        # Generate the raw response from the current thinking model
        result = ollama_gen_ans(
            curr_id,
            prompt,
            response_format=None,
        )
        s[f"{curr_name}_raw_output_{round}"] = result
        s[f"{curr_name}_output_{round}"] = summarize(summ_id, result)

    return all_results


def prepare_prompt(test_sample, round, history=None):
    prompt = []

    if round == 0:
        prompt.append("""
    You will be asked a causal reasoning question. You should structure your final answer as follows: Step 1) Extract the causal graph: Identify the causal graph that depicts the relationships in the scenario. The diagram should simply consist of edges denoted in "var1 -> var2" format, separated by commas.  Step 2) Determine the query type: Identify the type of query implied by the main question. Choices include "marginal probability", "conditional probability", "explaining away effect", "backdoor adjustment set", "average treatment effect", "collider bias", "normal counterfactual question", "average treatment effect on treated", "natural direct effect" or "natural indirect effect". Your answer should only be a term from the list above, enclosed in quotation marks.  Step 3) Formalize the query: Translate the query into its formal mathematical expression based on its type, utilizing the "do(·)" notation or counterfactual notations as needed.  Step 4) Gather all relevant data: Extract all the available data. Your answer should contain nothing but marginal probabilities and conditional probabilities in the form "P(...)=..." or "P(...|...)=...", each probability being separated by a semicolon. Stick to the previously mentioned denotations for the variables.  Step 5) Deduce the estimand using causal inference: Given all the information above, deduce the estimand using skills such as do-calculus, counterfactual prediction, and the basics of probabilities. Answer step by step.  Step 6) Calculate the estimand: Insert the relevant data in Step 4 into the estimand, perform basic arithmetic calculations, and derive the final answer. Step 7) Give a final yes/no answer to the question.
    There is an identifiable yes/no answer, which may sometimes go against your commonsense intuition. Be confident in your thinking: while answers may be unintuitive, there are no trick questions, and answers will be obvious once calculated.\n
    """)

    prompt.append(f"{test_sample['background']} \n {test_sample['given_info']} \n")
    prompt.append(f"{test_sample['question']} \n")

    if history:
        prompt.append("\n".join(history))

        if round % 2 != 0:
            prompt.append(
                "\nCarefully review your opponent's solution. If you identify any flaws in their reasoning, you should point them out and suggest corrections."
            )
        if round % 2 == 0:
            prompt.append(
                "\nYou may defend your previous answers, or be persuaded to your opponent's solution, as you see fit."
            )

    prompt.append(
        "\nYou should explicitly state your level of confidence in your answer (between 0.0 and 1.0)."
    )

    return "\n".join(prompt)
