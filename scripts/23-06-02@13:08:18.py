import transformers
import torch

from workshop import *


def stop_on_deindent(text):
    last_line = text.splitlines()[-1]
    if len(last_line) > 0 and last_line[0] not in (" ", "\t"):
        truncated = "\n".join(text.splitlines()[:-1])
        return False, truncated
    else:
        return True, None
    

class SampleBatch:

    def __init__(self, n_samples):
        self.samples = [""] * n_samples
        self.states = [True] * n_samples
        self.index_map = { n: n for n in range(n_samples) }
        self.accepted_indices = []

    def put_token(self, index, token):
        true_index = self.index_map[index]
        sample = self.samples[true_index]
        sample += token
        accepted, truncated = stop_on_deindent(sample)
        if accepted:
            self.samples[true_index] = sample
            self.index_map[len(self.accepted_indices)] = true_index
            self.accepted_indices.append(index)
        else:
            self.samples[true_index] = truncated
            self.states[true_index] = False

        return accepted

    def step(self):
        self.accepted_indices = []

    def n_running(self):
        return sum(self.states)

    def is_done(self):
        for i in range(len(self.states)):
            state = self.states[i]
            if state:
                return False
        return True


if __name__ == "__main__":
    model_uri = "Daoguang/PyCodeGPT"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_uri)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_uri)
    model.to("cuda:0")

    prompt = """\
# Return the largest number present in both lists
def largest_common(numbers1, numbers2):
"""

    n_samples = 4
    max_tokens = 1024

    inputs = tokenizer([prompt] * n_samples, return_tensors="pt").to("cuda:0")
    prompt_length = inputs["input_ids"].shape[1]
    n_generated_tokens = 0

    batch = SampleBatch(n_samples)
    running_sample_indices = list(range(n_samples))

    while prompt_length + n_generated_tokens <= max_tokens:
        outputs = model(**inputs)

        next_token_ids = []
        for i in range(batch.n_running()):
            logits = outputs.logits[i, -1]
            p = torch.distributions.Categorical(logits=logits)
            sampled_id = p.sample((1,))
            next_token = tokenizer.decode(sampled_id)

            accepted = batch.put_token(i, next_token)
            if accepted:
                next_token_ids.append([sampled_id])

        n_generated_tokens += 1

        if batch.is_done():
            break

        new_layers = []
        for layer in outputs.past_key_values:
            new_layers.append(
                (
                    layer[0][batch.accepted_indices],
                    layer[1][batch.accepted_indices]
                )
            )

        batch.step()

        inputs = {
            "input_ids": torch.tensor(next_token_ids).to("cuda:0"),
            "past_key_values": new_layers
        }

    for response in batch.samples:
        print(prompt, end="")
        print(response)

        print()
        print("-" * 80)
        print()