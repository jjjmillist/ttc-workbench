import transformers
import torch


def shape(structure):
    try:
        return structure.shape
    except AttributeError:
        return (f"list[{len(structure)}]", *shape(structure[0]))


short_prompt = """To be or not to"""

long_prompt = """It was the best of times, it was the worst"""
    

if __name__ == "__main__":
    print("Started")

    model_uri = "gpt2"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_uri)
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(model_uri)
    model.to("cuda:0")

    inputs = tokenizer([short_prompt, long_prompt], padding=True, return_tensors="pt").to("cuda:0")
    
    generated = []
    for _ in range(10):
        print(inputs.keys())
        print(shape(inputs["input_ids"]))
        print(shape(inputs["attention_mask"]))
        print(inputs["attention_mask"].sum(axis=1, dtype=torch.int64))
        with torch.no_grad():
            outputs = model(**inputs)

        print(shape(outputs.past_key_values))
        print()

        next_tokens = outputs.logits[:, -1, :].max(axis=-1).indices

        generated.append(tokenizer.decode(next_tokens))
        
        insertion_points = inputs["attention_mask"].sum(axis=1, dtype=torch.int64)
        new_column = torch.tensor(tokenizer.pad_token_id).repeat(2).to("cuda:0")    
        new_inputs = torch.cat((inputs["input_ids"], new_column[:, None]), dim=1)
        new_inputs.scatter_(1, insertion_points[:, None], next_tokens[:, None])

        mask = inputs["attention_mask"]
        new_mask_column = torch.zeros((len(inputs["input_ids"]), 1)).to("cuda:0")
        new_mask = torch.cat((mask, new_mask_column), dim=1)
        new_mask.scatter_(1, insertion_points[:, None], torch.ones(2, 1).to("cuda:0"))

        # inputs["input_ids"] = new_inputs
        # inputs["attention_mask"] = new_mask
        # inputs.past_key_values = outputs.past_key_values


        inputs = {
            "input_ids": next_tokens,
            # "attention_mask": new_mask,
            "past_key_values": outputs.past_key_values
        }