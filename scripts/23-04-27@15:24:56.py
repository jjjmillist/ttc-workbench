import transformers
import torch
from   transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
import pickle
import numpy as np

from data import *
from workshop import *


def stop_on_comment(text):
    last_line = text.splitlines()[-1]
    if len(last_line) > 0 and last_line[0] == "#":
        truncated = "\n".join(text.splitlines()[:-1])
        return False, truncated
    else:
        return True, None


def predict(model, tokenizer, prompt, stopping_strategy=None, k=10, batch_size=10):
    generated_text = ""
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt").to("cuda:0")
    prompt_length = inputs["input_ids"].shape[1]
    n_generated_tokens = 0
    context_size = model.config.n_ctx

    # print(f"Running prediction on a prompt with {prompt_length} tokens")

    generated_strings = [""] * batch_size
    running_sample_indices = list(range(batch_size))

    while prompt_length + n_generated_tokens <= context_size:
        # print("  Generating token")
        # print("    Running samples:", running_sample_indices)

        outputs = model(**inputs)
        top_k = outputs.logits[:, -1].topk(k)

        next_token_ids = []
        accepted_sample_indices = []
        for i in range(len(running_sample_indices)):
            # print(f"      Extending sample number {i}: {running_sample_indices[i]}")

            logits = top_k.values[i]
            p = torch.distributions.Categorical(logits=logits)
            j = p.sample((1,))
            sampled_id = top_k.indices[i, j]
            next_token = tokenizer.decode(sampled_id)

            # print(f"        Chose {sampled_id[0]}: {next_token}")

            sample_index = running_sample_indices[i]
            generated_strings[sample_index] += next_token
        
            accepted, truncated = stopping_strategy(generated_strings[sample_index])
            if not accepted:
                generated_strings[sample_index] = truncated                
                # print(f"        Sample is finished")
            else:
                next_token_ids.append([sampled_id])
                accepted_sample_indices.append(i)
                # print(f"        Sample continues")

        running_sample_indices = [running_sample_indices[i] for i in accepted_sample_indices]

        # print("    Accepted:", accepted_sample_indices)
        # print("    Running samples:", running_sample_indices)

        n_generated_tokens += 1

        if len(running_sample_indices) == 0:
            break

        new_layers = []
        for layer in outputs.past_key_values:
            new_layers.append(
                (
                    layer[0][accepted_sample_indices],
                    layer[1][accepted_sample_indices]
                )
            )

        inputs = {
            "input_ids": torch.tensor(next_token_ids).to("cuda:0"),
            "past_key_values": new_layers
        }

    return generated_strings

root = output_directory("codeparrot-mbpp")

tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot")
model = AutoModelForCausalLM.from_pretrained("codeparrot/codeparrot")
model.to("cuda:0")

k = 3
mbpp = mbpp_sanitized()

seeds = list(range(30))
test_indices = list(range(len(mbpp)))
batch_size = 20

t0 = time()
for seed in seeds:
    with torch.no_grad():
        rng = np.random.default_rng(seed)
        writers = [OutputWriter(root / f"seed_{seed}" / f"output_{n}") for n in range(batch_size)]
        prompts = []

        for test_index in test_indices:
            test_instance = mbpp[test_index]
            train_indices = rng.choice(len(mbpp), size=3, replace=False)
            training_instances = [mbpp[i] for i in train_indices]

            prefix_and_prompt = few_shot_mbpp(test_instance, training_instances)
            
            prompts.append(prefix_and_prompt)

        with open(root / f"seed_{seed}_prompts.pickle", "wb") as prompt_file:
            pickle.dump(prompts, prompt_file)
        
        for prompt in prompts:
            responses = predict(
                model,
                tokenizer,
                prompt,
                stopping_strategy=stop_on_comment,
                k=10,
                batch_size=batch_size
            )

            for text, writer in zip(responses, writers):
                writer.write(text)