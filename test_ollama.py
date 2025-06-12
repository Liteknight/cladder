import ollama
from ollama import chat
import json
import pandas as pd

# client = ollama.Client()

file1 = open("/home/finn/Documents/Python/cladder/data/cladder-v1/cladder-v1-q-balanced.json")
file2 = open("/home/finn/Documents/Python/cladder/data/cladder-v1-meta-models.json")

data = json.load(file1)
meta = json.load(file2)

df = pd.DataFrame(data)
df2 = pd.DataFrame(meta)

# Extract only 'rung' and 'model_id' from 'meta'
df['rung'] = df['meta'].apply(lambda x: x.get('rung'))
df['model_id'] = df['meta'].apply(lambda x: x.get('model_id'))

# Build a lookup dictionary for model_id -> background
model_backgrounds = {model['model_id']: model['background'] for model in df2.to_dict(orient="records")}
df['background'] = df['model_id'].map(model_backgrounds)

df = df.drop(columns=['meta'])

# print(df.head())
# print(df.columns.values)

model = "qwen3:32b"
# model = "deepseek-r1"
# model = "gemma3:27b"

guide_normal = """
    You are an expert in causal inference. The following question is not a typical commonsense query, but rather a meticulously designed question created by a professor specializing in causal inference, intended to assess the students' mastery of the course content. There may be unneccessary information included to try to distract you, so you must ignore what you believe is not relevant.
You should structure your asnwer as follows: Step 1) Extract the causal graph: Identify the causal graph that depicts the relationships in the scenario. The diagram should simply consist of edges denoted in "var1 -> var2" format, separated by commas.  Step 2) Determine the query type: Identify the type of query implied by the main question. Choices include "marginal probability", "conditional probability", "explaining away effect", "backdoor adjustment set", "average treatment effect", "collider bias", "normal counterfactual question", "average treatment effect on treated", "natural direct effect" or "natural indirect effect". Your answer should only be a term from the list above, enclosed in quotation marks.  Step 3) Formalize the query: Translate the query into its formal mathematical expression based on its type, utilizing the "do(·)" notation or counterfactual notations as needed.  Step 4) Gather all relevant data: Extract all the available data. Your answer should contain nothing but marginal probabilities and conditional probabilities in the form "P(...)=..." or "P(...|...)=...", each probability being separated by a semicolon. Stick to the previously mentioned denotations for the variables.  Step 5) Deduce the estimand using causal inference: Given all the information above, deduce the estimand using skills such as do-calculus, counterfactual prediction, and the basics of probabilities. Answer step by step.  Step 6) Calculate the estimand: Insert the relevant data in Step 4 into the estimand, perform basic arithmetic calculations, and derive the final answer. Step 7) Give a final yes/no answer to the question.
There is an identifiable yes/no answer. Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most.
You are VERY knowledgeable. An unparalleled expert. Think and respond with UTMOST confidence.
"""

guide_think = """
You should output you Step 1) Extract the causal graph: Identify the causal graph that depicts the relationships in the scenario. The diagram should simply consist of edges denoted in "var1 -> var2" format, separated by commas.  Step 2) Determine the query type: Identify the type of query implied by the main question. Choices include "marginal probability", "conditional probability", "explaining away effect", "backdoor adjustment set", "average treatment effect", "collider bias", "normal counterfactual question", "average treatment effect on treated", "natural direct effect" or "natural indirect effect". Your answer should only be a term from the list above, enclosed in quotation marks.  Step 3) Formalize the query: Translate the query into its formal mathematical expression based on its type, utilizing the "do(·)" notation or counterfactual notations as needed.  Step 4) Gather all relevant data: Extract all the available data. Your answer should contain nothing but marginal probabilities and conditional probabilities in the form "P(...)=..." or "P(...|...)=...", each probability being separated by a semicolon. Stick to the previously mentioned denotations for the variables.  Step 5) Deduce the estimand using causal inference: Given all the information above, deduce the estimand using skills such as do-calculus, counterfactual prediction, and the basics of probabilities. Answer step by step.  Step 6) Calculate the estimand: Insert the relevant data in Step 4 into the estimand, perform basic arithmetic calculations, and derive the final answer. Step 7) Give a final yes/no answer to the question.
There is an identifiable yes/no answer. Answer step by step, where each thinking step has AT MOST 20 words -- in other words, you need to be concise in your reasoning trace.
You are VERY knowledgeable. An unparalleled expert. Think and respond with UTMOST confidence. DO NOT OVERTHINK PLEASE.
"""

# question 73
# backgrnd = """
# "Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: CEO has a direct effect on director and manager. Manager has a direct effect on employee. Director has a direct effect on employee."
# """
# given = """
# "The overall probability of manager signing the termination letter is 86%. For managers who don't sign termination letters, the probability of employee being fired is 31%. For managers who sign termination letters, the probability of employee being fired is 70%."
# """
# question = """
# "Is employee being fired more likely than employee not being fired overall?"
# """

idx = 73

backgrnd = df.iloc[idx]["background"]
given = df.iloc[idx]["given_info"]
question = df.iloc[idx]["question"]
answer = df.iloc[idx]["answer"]

prompt = backgrnd + given + question

print(prompt)
print(answer)

stream = chat(
    model=model,
    messages=[
        {"role": "system", "content": guide_think},
        {"role": "user", "content": prompt},
    ],
    stream=True,
    options={"temperature": 0.6, "top_p": 0.95},
    think=False,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
