import pickle
import numpy as np
from data import mbpp_sanitized
from workshop import *
import evaluate

root = output_directory("perplexity")

perplexity = evaluate.load("perplexity", module_type="measurement")

inputs = []

mbpp = list(mbpp_sanitized())
n_data = len(mbpp)
for prefix_index in range(n_data):
    for test_index in range(n_data):
        prefix_and_prompt = ""
        for line in mbpp[prefix_index]["prompt"].splitlines():
            prefix_and_prompt += "# " + line + "\n"
        prefix_and_prompt += mbpp[prefix_index]["code"] + "\n\n"

        for line in mbpp[test_index]["prompt"].splitlines():
            prefix_and_prompt += "# " + line + "\n"
        for line in mbpp[test_index]["code"].splitlines():
            if line.startswith("def "):
                prefix_and_prompt += line.strip() + "\n"
                break
        
        inputs.append(prefix_and_prompt)

perplexities = perplexity.compute(
    data=inputs,
    model_id="codeparrot/codeparrot",
    add_start_token=False
)["perplexities"]

perplexity_matrix = np.empty((n_data, n_data))
for prefix_index in range(n_data):
    for test_index in range(n_data):
        offset = prefix_index * n_data + test_index
        perplexity_matrix[prefix_index, test_index] = perplexities[offset]

results = {
    "perplexities": perplexities,
    "inputs": inputs
}

with open(root / "perplexities", "wb") as file:
    pickle.dump(results, file)