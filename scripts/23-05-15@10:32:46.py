import pickle
import transformers
import torch
from timer import TimedInterval
import data
from workshop import *


def print_line(symbol="-"):
    print()
    print(symbol * 100)
    print()


def never_stop(text):
    return True, None

    
def stop_on_deindent(text):
    last_line = text.splitlines()[-1]
    if len(last_line) > 0 and last_line[0] not in (" ", "\t"):
        truncated = "\n".join(text.splitlines()[:-1])
        return False, truncated
    else:
        return True, None
    

def old_predict(model, tokenizer, prompt, timer, stopping_strategy=None, k=10, batch_size=10, max_tokens=1024):
    generated_text = ""
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt").to("cuda:0")
    prompt_length = inputs["input_ids"].shape[1]
    n_generated_tokens = 0
    context_size = max_tokens

    # print(f"Running prediction on a prompt with {prompt_length} tokens")

    generated_strings = [""] * batch_size
    running_sample_indices = list(range(batch_size))

    while prompt_length + n_generated_tokens <= context_size:
        # print("  Generating token")
        # print("    Running samples:", running_sample_indices)

        print(prompt_length + n_generated_tokens)

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


def predict(model, tokenizer, prompt, timer, stopping_strategy=None, k=10, batch_size=10, max_tokens=1024):
    if torch.is_grad_enabled():
        print("WARNING! You are running prediction without no_grad(). You are likely to run out of memory.")

    torch.manual_seed(0)

    inputs = tokenizer([prompt] * batch_size, return_tensors="pt").to("cuda:0")
    prompt_length = inputs["input_ids"].shape[1]
    n_generated_tokens = 0

    generated_strings = [""] * batch_size
    running_sample_indices = list(range(batch_size))
        
    while prompt_length + n_generated_tokens <= max_tokens:
        token_timer = timer.subinterval("token")
        with token_timer:
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)            
            token_timer.put("reserved_bytes_before", reserved)
            token_timer.put("allocated_bytes_before", allocated)
            token_timer.put("context_size", prompt_length + n_generated_tokens)
            token_timer.put("batch_size", len(running_sample_indices))

            print(prompt_length + n_generated_tokens)
            forward_pass_timer = token_timer.subinterval("forward")
            with forward_pass_timer:
                outputs = model(**inputs)

            top_k = outputs.logits[:, -1].topk(k)

            next_token_ids = []
            accepted_sample_indices = []
            for i in range(len(running_sample_indices)):
                logits = top_k.values[i]
                p = torch.distributions.Categorical(logits=logits)
                j = p.sample((1,))
                sampled_id = top_k.indices[i, j]
                next_token = tokenizer.decode(sampled_id)

                sample_index = running_sample_indices[i]
                generated_strings[sample_index] += next_token
            
                accepted, truncated = stopping_strategy(generated_strings[sample_index])
                if not accepted:
                    generated_strings[sample_index] = truncated
                else:
                    next_token_ids.append([sampled_id])
                    accepted_sample_indices.append(i)

            running_sample_indices = [running_sample_indices[i] for i in accepted_sample_indices]

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


run_timer = TimedInterval()


if __name__ == "__main__":
    print("Started")

    model_uri = "Daoguang/PyCodeGPT"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_uri)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_uri)
    model.to("cuda:0")

    prompts = list(data.mbpp_zero_shot())

    batch_size = 20

    # root = output_directory()
    # writers = [OutputWriter(root / f"output_{n}") for n in range(batch_size)]
    # prompt_writer = OutputWriter(root / "prompts")
    
    n_prompts = len(prompts)
    n_done = 0

    hello_prompt = """\
def hello_world():

"""

    codeparrot_prompt = """\
# Write a python function to find the length of the longest sublists.
def Find_Max_Length(lst):  
    maxLength = max(len(x) for x in lst )
    return maxLength 

# Write a function to find the directrix of a parabola.
def parabola_directrix(a, b, c): 
  directrix=((int)(c - ((b * b) + 1) * 4 * a ))
  return directrix

# Write a python function to find the index of smallest triangular number with n digits. https://www.geeksforgeeks.org/index-of-smallest-triangular-number-with-n-digits/
import math 
def find_Index(n): 
    x = math.sqrt(2 * math.pow(10,(n - 1)))
    return round(x)

# Write a function to find the shared elements from the given two lists.
def similar_elements(test_tup1, test_tup2):

"""

    seconds_so_far = 0
    bad = 198
    with torch.no_grad():
        with run_timer:
            for prompt in [hello_prompt]:
                batch_timer = run_timer.subinterval("predict")
                batch_timer.put("prompt", prompt)
                with batch_timer:
                    responses = predict(
                        model,
                        tokenizer,
                        prompt,
                        stopping_strategy=never_stop,
                        k=10,
                        batch_size=batch_size,
                        timer=batch_timer,
                        max_tokens=1024
                    )

                # for snippet, writer in zip(responses, writers):
                #     writer.write(snippet)

                n_done += 1
                seconds_so_far += batch_timer.duration
                secs_per_prompt = seconds_so_far / n_done
                eta = ((n_prompts - n_done) * secs_per_prompt) / 60
                print(f"{n_done}/{n_prompts} - {secs_per_prompt:.2f}spp - eta +{eta:.2f}m")

    # with open(root / "timing.pickle", "wb") as file:
    #     pickle.dump(run_timer, file)