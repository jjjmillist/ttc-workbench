import transformers
import numpy as np
import pickle
from pathlib import Path
from workshop import *

if __name__ == "__main__":
    root = Path("results/codeparrot-mbpp-28-04-23@20:50:35")
    all_snippets = []
    for i in range(30):
        with open(root / f"seed_{i}_prompts.pickle", "rb") as file:
            prompts = pickle.load(file)
            
        for j in range(20):
            filepath = root / f"seed_{i}" / f"output_{j}"
            with open(filepath, "r") as file:
                snippets = read_output_file(filepath)
                all_snippets += [p + s for p, s in zip(prompts, snippets)]

    tokenizer = transformers.AutoTokenizer.from_pretrained("codeparrot/codeparrot")
    result = tokenizer(all_snippets[:1], padding=False)

    print(all_snippets[0])
